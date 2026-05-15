# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
import multiprocessing
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParserManager
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.utils.import_utils import LazyLoader
from vllm.v1.structured_output.backend_guidance import GuidanceBackend
from vllm.v1.structured_output.backend_types import (
    StructuredOutputBackend,
    StructuredOutputGrammar,
)
from vllm.v1.structured_output.backend_xgrammar import XgrammarBackend

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import torch

    from vllm.reasoning import ReasoningParser
    from vllm.v1.request import Request
else:
    torch = LazyLoader("torch", globals(), "torch")


logger = init_logger(__name__)


class StructuredOutputManager:
    """Engine-level manager for structured output requests."""

    def __init__(self, vllm_config: VllmConfig):
        self.backend: StructuredOutputBackend | None = None
        # We only store the class of the reasoner in the manager.
        # The parser instance is request-scoped because some reasoning parsers
        # depend on per-request chat-template kwargs.
        self.reasoner_cls: type[ReasoningParser] | None = None
        self.vllm_config = vllm_config

        # When in external_launcher mode, async grammar compilation causes deadlocks
        # due to external_launcher mode having a scheduler for each TP rank.
        # Async grammar compilation causes the
        # WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR → WAITING transition to
        # happen at different times on different TP ranks,
        # breaking the determinism assumption that external_launcher relies on.
        self._use_async_grammar_compilation = (
            vllm_config.parallel_config.distributed_executor_backend
            != "external_launcher"
        )

        self._grammar_bitmask: torch.Tensor | None = None
        self._full_mask = torch.tensor(-1, dtype=torch.int32)

        max_batch_size = self.vllm_config.scheduler_config.max_num_seqs
        self.fill_bitmask_parallel_threshold = 128
        if self.fill_bitmask_parallel_threshold < max_batch_size:
            self.fill_bitmask_parallel_batch_size = 16
            # Use:
            # - at least 1 CPU
            # - at most half the number of CPUs or 8, whichever is less
            max_workers = max(1, min(multiprocessing.cpu_count() // 2, 8))
            self.executor_for_fillmask = ThreadPoolExecutor(max_workers=max_workers)

        if not self.vllm_config.model_config.skip_tokenizer_init:
            # The default max_workers if not specified is the number of
            # CPUs * 5, which is way too high since these tasks are CPU-bound,
            # not I/O bound. We also know we would never dominate CPU usage
            # with just grammar compilation, so we set it to half the number
            # of CPUs.
            max_workers = max(1, (multiprocessing.cpu_count() + 1) // 2)
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
            self.tokenizer = cached_tokenizer_from_config(
                model_config=self.vllm_config.model_config
            )
            reasoning_parser_plugin = (
                self.vllm_config.structured_outputs_config.reasoning_parser_plugin
            )
            if reasoning_parser_plugin and len(reasoning_parser_plugin) > 3:
                ReasoningParserManager.import_reasoning_parser(reasoning_parser_plugin)

            reasoning_parser = (
                self.vllm_config.structured_outputs_config.reasoning_parser
            )
            if reasoning_parser:
                self.reasoner_cls = ReasoningParserManager.get_reasoning_parser(
                    reasoning_parser
                )

        self.enable_in_reasoning = (
            self.vllm_config.structured_outputs_config.enable_in_reasoning
        )

    def _get_reasoner(self, request: "Request") -> "ReasoningParser | None":
        structured_req = request.structured_output_request
        if structured_req is None or self.reasoner_cls is None:
            return None

        if structured_req.reasoner is None:
            # Lazily build the request-local parser so the structured-output
            # gate observes the same template kwargs used by the frontend.
            parser_kwargs = structured_req.reasoning_parser_kwargs or {}
            structured_req.reasoner = self.reasoner_cls(
                tokenizer=self.tokenizer,
                **parser_kwargs,
            )
        return structured_req.reasoner

    def grammar_init(self, request: "Request") -> None:
        if request.structured_output_request is None:
            return

        if TYPE_CHECKING:
            assert (
                request.sampling_params is not None
                and request.sampling_params.structured_outputs is not None
            )

        # Initialize the backend the first time it is needed.
        #
        # NOTE: We only support a single backend. We do NOT support different
        # backends on a per-request basis in V1 (for now, anyway...).
        # _backend is set in Processor._validate_structured_output
        if self.backend is None:
            assert request.sampling_params is not None
            backend = request.sampling_params.structured_outputs._backend
            vocab_size = self.vllm_config.model_config.get_vocab_size()
            if backend == "xgrammar":
                self.backend = XgrammarBackend(
                    self.vllm_config,
                    tokenizer=self.tokenizer,
                    vocab_size=vocab_size,
                )
            elif backend == "guidance":
                self.backend = GuidanceBackend(
                    self.vllm_config,
                    tokenizer=self.tokenizer,
                    vocab_size=vocab_size,
                )
            elif backend == "outlines":
                from vllm.v1.structured_output.backend_outlines import OutlinesBackend

                self.backend = OutlinesBackend(
                    self.vllm_config,
                    tokenizer=self.tokenizer,
                    vocab_size=vocab_size,
                )
            elif backend == "lm-format-enforcer":
                from vllm.v1.structured_output.backend_lm_format_enforcer import (  # noqa: E501
                    LMFormatEnforcerBackend,
                )

                self.backend = LMFormatEnforcerBackend(
                    self.vllm_config,
                    tokenizer=self.tokenizer,
                    vocab_size=vocab_size,
                )
            else:
                raise ValueError(f"Unsupported structured output backend: {backend}")

        if self._use_async_grammar_compilation:
            grammar = self.executor.submit(self._create_grammar, request)
        else:
            grammar = self._create_grammar(request)  # type: ignore[assignment]
        request.structured_output_request.grammar = grammar  # type: ignore[assignment]

    def _create_grammar(self, request: "Request") -> StructuredOutputGrammar:
        key = request.structured_output_request.structured_output_key  # type: ignore[union-attr]

        # Note that the request was validated in the engine core client,
        # so at this point we know it is a supported type of request.
        #
        # TODO: we still need to handle xgrammar compilation failures,
        # though it should be unlikely as we test that up front as well.
        request_type, grammar_spec = key

        assert self.backend is not None
        return self.backend.compile_grammar(request_type, grammar_spec)

    def _fill_bitmasks(
        self, batch: Iterable[tuple[StructuredOutputGrammar, int, bool]]
    ) -> None:
        assert self._grammar_bitmask is not None
        for grammar, index, apply_bitmask in batch:
            if apply_bitmask and not grammar.is_terminated():
                grammar.fill_bitmask(self._grammar_bitmask, index)
            else:
                # Note that for thinking support, we will need to
                # reset the relevant part of the bitmask for consequent
                # requests here.
                self._grammar_bitmask[index].fill_(self._full_mask)

    def _init_reasoning_ended(
        self,
        request: "Request",
        reasoner: "ReasoningParser",
    ) -> bool:
        structured_req = request.structured_output_request
        assert structured_req is not None
        if structured_req.reasoning_ended is None:
            # Preserve the existing prompt-level initialization. Some clients
            # pass a request that has already exited reasoning before decode
            # starts, and those requests should be constrained immediately.
            structured_req.reasoning_ended = reasoner.is_reasoning_end(
                request.prompt_token_ids or []
            )
        return structured_req.reasoning_ended

    def _find_reasoning_end_in_tokens(
        self,
        request: "Request",
        token_ids: list[int],
        *,
        tokens_already_appended: bool,
    ) -> int | None:
        """Return the offset of the reasoning-end token inside token_ids.

        MTP can accept a batch containing both the reasoning-end marker and
        visible answer tokens after it. The old boolean gate only noticed that
        reasoning had ended, so those visible suffix tokens escaped xgrammar.
        This scans the small per-step token list so callers can split:
        - unconstrained reasoning prefix through the end marker, and
        - constrained visible suffix after the marker.
        """
        if not token_ids:
            return None

        reasoner = self._get_reasoner(request)
        if reasoner is None or self.enable_in_reasoning:
            return None

        structured_req = request.structured_output_request
        assert structured_req is not None
        if self._init_reasoning_ended(request, reasoner):
            return None

        all_token_ids = request.all_token_ids
        base_token_ids = all_token_ids
        if tokens_already_appended:
            if (
                len(token_ids) <= len(all_token_ids)
                and all_token_ids[-len(token_ids) :] == token_ids
            ):
                base_token_ids = all_token_ids[: -len(token_ids)]
            else:
                logger.debug(
                    "Structured output reasoning gate expected request %s to end "
                    "with generated tokens %s, but all_token_ids ended with %s.",
                    request.request_id,
                    token_ids,
                    all_token_ids[-len(token_ids) :] if token_ids else [],
                )

        for index in range(len(token_ids)):
            delta_ids = token_ids[: index + 1]
            candidate_token_ids = [*base_token_ids, *delta_ids]
            if reasoner.is_reasoning_end_streaming(candidate_token_ids, delta_ids):
                return index
        return None

    def identify_constrained_token_ids(
        self,
        request: "Request",
        token_ids: list[int],
        *,
        tokens_already_appended: bool = False,
    ) -> tuple[list[int], list[int]]:
        """Split token_ids into unconstrained reasoning and constrained content."""
        if not token_ids:
            return [], []
        if not request.use_structured_output:
            return token_ids, []

        reasoner = self._get_reasoner(request)
        if reasoner is None or self.enable_in_reasoning:
            return [], token_ids

        if self._init_reasoning_ended(request, reasoner):
            return [], token_ids

        reasoning_end_index = self._find_reasoning_end_in_tokens(
            request, token_ids, tokens_already_appended=tokens_already_appended
        )
        if reasoning_end_index is None:
            return token_ids, []

        first_constrained_index = reasoning_end_index + 1
        return token_ids[:first_constrained_index], token_ids[first_constrained_index:]

    def get_grammar_advance_token_ids(
        self, request: "Request", new_token_ids: list[int]
    ) -> list[int]:
        """Return only the visible tokens that should advance the grammar.

        This is called after target-model output has been appended, so this is the
        one place where detecting a reasoning-end marker should commit
        reasoning_ended=True.
        """
        if not request.use_structured_output:
            return []

        reasoner = self._get_reasoner(request)
        if reasoner is None or self.enable_in_reasoning:
            return new_token_ids

        structured_req = request.structured_output_request
        assert structured_req is not None

        if self._init_reasoning_ended(request, reasoner):
            return new_token_ids

        reasoning_end_index = self._find_reasoning_end_in_tokens(
            request,
            new_token_ids,
            tokens_already_appended=True,
        )
        if reasoning_end_index is None:
            return []

        structured_req.reasoning_ended = True
        return new_token_ids[reasoning_end_index + 1 :]

    def validate_tokens_reasoning_aware(
        self, request: "Request", token_ids: list[int]
    ) -> list[int]:
        """Validate speculative tokens without constraining reasoning tokens."""
        unconstrained_token_ids, constrained_token_ids = (
            self.identify_constrained_token_ids(request, token_ids)
        )
        if not constrained_token_ids:
            return unconstrained_token_ids

        structured_req = request.structured_output_request
        assert structured_req is not None and structured_req.grammar is not None
        validated_token_ids = structured_req.grammar.validate_tokens(
            constrained_token_ids
        )
        if len(validated_token_ids) != len(constrained_token_ids):
            logger.debug(
                "Structured output grammar trimmed speculative visible tokens "
                "for request %s: original suffix=%s validated suffix=%s.",
                request.request_id,
                constrained_token_ids,
                validated_token_ids,
            )
        return [*unconstrained_token_ids, *validated_token_ids]

    def _async_submit_fill_bitmask(
        self, batch: list[tuple[StructuredOutputGrammar, int, bool]]
    ) -> Future:
        return self.executor_for_fillmask.submit(self._fill_bitmasks, batch)

    def grammar_bitmask(
        self,
        requests: dict[str, "Request"],
        structured_output_request_ids: list[str],
        scheduled_spec_decode_tokens: dict[str, list[int]],
    ) -> "npt.NDArray[np.int32] | None":
        # Prepare the structured output bitmask for this batch.
        if not structured_output_request_ids:
            return None

        max_num_spec_tokens = 0
        if self.vllm_config.speculative_config is not None:
            max_num_spec_tokens = (
                self.vllm_config.speculative_config.num_speculative_tokens
            )

        if self._grammar_bitmask is None:
            assert self.backend is not None
            max_batch_size = self.vllm_config.scheduler_config.max_num_seqs

            # Allocate a bitmask for each token needing to be checked:
            # one for each speculative position, and one more for the
            # bonus token / non-speculative token.
            self._grammar_bitmask = self.backend.allocate_token_bitmask(
                max_batch_size * (1 + max_num_spec_tokens)
            )

        # Generate a batched bitmask for all structured output requests.
        # When speculative decoding is enabled, we need to include multiple
        # masks for each request, one for each possible bonus token position.
        # These are stored inline in the tensor and unpacked by the gpu runner.
        cumulative_index = 0

        # Optimized parallel filling of bitmasks for
        # non-spec, large-batch-size cases
        if (
            len(structured_output_request_ids) > self.fill_bitmask_parallel_threshold
            and max_num_spec_tokens == 0
        ):
            promises = []
            batch = []
            for req_id in structured_output_request_ids:
                request = requests[req_id]
                structured_output_request = request.structured_output_request
                if TYPE_CHECKING:
                    assert structured_output_request is not None
                    assert structured_output_request.grammar is not None
                grammar = structured_output_request.grammar

                apply_bitmask = self.should_fill_bitmask(request)
                batch.append((grammar, cumulative_index, apply_bitmask))
                if len(batch) == self.fill_bitmask_parallel_batch_size:
                    promises.append(self._async_submit_fill_bitmask(batch))
                    batch = []

                cumulative_index += 1
            if batch:
                promises.append(self._async_submit_fill_bitmask(batch))

            # Wait for all bitmask filling tasks to complete.
            for promise in promises:
                promise.result()
        else:
            # Fallback to serial filling of bitmasks for small-batch-size cases
            for req_id in structured_output_request_ids:
                request = requests[req_id]
                structured_output_request = request.structured_output_request

                if TYPE_CHECKING:
                    assert structured_output_request is not None
                    assert structured_output_request.grammar is not None
                grammar = structured_output_request.grammar
                req_tokens = list(scheduled_spec_decode_tokens.get(req_id, ()))

                # In MTP, req_tokens have not been appended yet. If reasoning
                # ends inside them, rows before and including the end token stay
                # unmasked; later draft rows and the bonus row use xgrammar.
                apply_bitmask = self.should_fill_bitmask(request)
                reasoning_end_index = None
                if req_tokens and not apply_bitmask:
                    reasoning_end_index = self._find_reasoning_end_in_tokens(
                        request, req_tokens, tokens_already_appended=False
                    )
                    if reasoning_end_index is not None:
                        apply_bitmask = True

                state_advancements = 0
                for spec_index, token in enumerate(itertools.chain(req_tokens, (-1,))):
                    row_apply_bitmask = apply_bitmask
                    if reasoning_end_index is not None:
                        row_apply_bitmask = spec_index > reasoning_end_index

                    self._fill_bitmasks(
                        ((grammar, cumulative_index, row_apply_bitmask),)
                    )
                    if token != -1 and row_apply_bitmask and not grammar.is_terminated():
                        accepted = grammar.accept_tokens(req_id, [token])
                        assert accepted, (token, req_id, scheduled_spec_decode_tokens)
                        state_advancements += 1
                    cumulative_index += 1
                if state_advancements > 0:
                    grammar.rollback(state_advancements)

        bitmask_tensor = self._grammar_bitmask
        if cumulative_index < bitmask_tensor.shape[0]:
            bitmask_tensor = bitmask_tensor[:cumulative_index]

        # After finishing with the xgrammar operations, we convert to
        # np.ndarray, because that is much more efficient for serialization
        # and deserialization when sending this to the GPU workers.
        return bitmask_tensor.numpy()

    def should_fill_bitmask(self, request: "Request") -> bool:
        # NOTE (Hanchen) if enable_in_reasoning is True, it means that
        # the model needs to be constrained in reasoning. So we should always
        # enable the bitmask filling.
        reasoner = self._get_reasoner(request)
        if reasoner is not None:
            if self.enable_in_reasoning:
                return True
            return self._init_reasoning_ended(request, reasoner)
        return True

    def should_advance(self, request: "Request", new_token_ids: list[int] | None = None,) -> bool:
        if not request.use_structured_output:
            return False

        # To determine whether we can advance the FSM.
        # Supports thinking usage where we skip the reasoning components.
        if TYPE_CHECKING:
            assert request.structured_output_request is not None
            assert request.structured_output_request.grammar is not None
        # by default, we should always advance
        # for cases that don't use thinking mode.
        reasoner = self._get_reasoner(request)
        if reasoner is None:
            return True

        # if the model needs structured in reasoning, we should advance
        if self.enable_in_reasoning:
            return True

        if self._init_reasoning_ended(request, reasoner):
            return True

        # Check if reasoning ends in *this* step
        if new_token_ids is None:
            delta_from = request.num_computed_tokens - request.num_output_placeholders
            all_token_ids = request.all_token_ids
            start = (
                delta_from
                if delta_from >= 0
                else max(len(all_token_ids) + delta_from, 0)
            )
            new_token_ids = list(itertools.islice(all_token_ids, start, None))
        reasoning_end_index = self._find_reasoning_end_in_tokens(
            request,
            new_token_ids,
            tokens_already_appended=True,
        )
        if reasoning_end_index is not None:
            request.structured_output_request.reasoning_ended = True

        return False

    def clear_backend(self) -> None:
        if self.backend is not None:
            self.backend.destroy()
