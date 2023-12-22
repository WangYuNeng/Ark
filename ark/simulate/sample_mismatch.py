from dataclasses import dataclass
from typing import Self

import jax.numpy as jnp
import jax.random as jrandom
import jax
from jax_dataclasses import pytree_dataclass, Static
from functools import reduce, partial, lru_cache

from ark.cdg.cdg import CDG, CDGNode, CDGEdge
from ark.specification.attribute_def import AttrDef, AttrDefMismatch
from ark.specification.specification import CDGSpec
import re


re_item_name = re.compile(r"^(\w+)_[^_]+$")
re_note_attr = re.compile(r"_([^_]+)$")


@lru_cache(maxsize=1)
def _get_node_edge_info(graph: CDG) -> tuple[dict, dict]:
    node_info = {n.name: n for n in graph.nodes}
    edge_info = {e.name: e for e in graph.edges}
    return node_info, edge_info


def get_attr_spec(graph: CDG, variable_name: str) -> AttrDef:
    node_info, edge_info = _get_node_edge_info(graph)
    graph_info = node_info | edge_info

    node_attr = re_note_attr.search(variable_name).group(1)
    item_name = re_item_name.match(variable_name).group(1)
    attr_def = graph_info[item_name].attr_def[node_attr]
    return attr_def


@dataclass
class SampleSpec:
    variable_dims: dict[str, int | None]
    counts: list[int]
    mismatch_count: int

    @classmethod
    def create(
        cls,
        variable_dims: dict[str, int | None],
        counts: list[int] | int,
        mismatch_count: int = 1,
    ) -> Self:
        if isinstance(counts, int):
            counts = [counts]
        assert len(counts) == max(variable_dims.values()) + 1
        return SampleSpec(
            variable_dims=variable_dims, counts=counts, mismatch_count=mismatch_count
        )


@pytree_dataclass
class GraphSample:
    sample_spec: Static[SampleSpec]
    attr_decision_samples: dict[str, jax.Array]
    attr_mismatch_samples: dict[str, jax.Array]


@dataclass
class ParameterSet:
    spec: CDGSpec
    cdg: CDG

    def sample_graph(self, *, sample_spec: SampleSpec, key: jax.Array) -> GraphSample:
        """
        Sample multiple graph configurations given a specification of which variables to map to which dimensions,
        and how many samples to draw in each dimension. Mismatch sampling will be in the first dimension if enabled
        (with mismatch_count > 1).
        """

        total_samples: int = reduce(lambda x, y: x * y, sample_spec.counts)
        print(
            f"Sampling {sample_spec.counts} (total: {total_samples}) according to {sample_spec}"
        )

        # Sample decision variables

        attr_specs = {
            var: get_attr_spec(self.cdg, var)
            for var in sample_spec.variable_dims.keys()
        }

        *decision_keys, key = jrandom.split(key, len(attr_specs) + 1)

        attr_decision_samples = {}
        attr_mismatch_samples = {}

        for decision_key, var in zip(decision_keys, attr_specs.keys()):
            attr_spec: AttrDef = attr_specs[var]
            attr_dimension = sample_spec.variable_dims[var]
            sample_count = sample_spec.counts[attr_dimension]

            print(f"Sampling {var} according to {attr_spec}")
            if isinstance(attr_spec, AttrDefMismatch):
                decision_key, mismatch_key = jrandom.split(decision_key)
                mismatch_tensor = attr_spec.sample_mismatch_ratio(
                    shape=(sample_spec.mismatch_count,), key=decision_key
                )
                attr_mismatch_samples[var] = mismatch_tensor
            decision_tensor = attr_spec.sample_decision(
                shape=(sample_count,), key=decision_key
            )
            attr_decision_samples[var] = decision_tensor

        return GraphSample(
            sample_spec=sample_spec,
            attr_decision_samples=attr_decision_samples,
            attr_mismatch_samples=attr_mismatch_samples,
        )
