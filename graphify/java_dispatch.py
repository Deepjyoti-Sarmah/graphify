"""Member-level interface dispatch for Java (#5319).

`this.worker.run()` on a field typed `Worker` resolves to the interface's own
`.run()`, because that is what the call site names. The implementing
`EmailWorker.run()` is a separate node, and nothing joins the two, so a
directed walk stops at the interface: the Java twin of the C# gap #3003 closed.

The mechanism lives in `graphify.interface_dispatch` and is shared with the
C#, Swift and Go passes; this module fixes the Java gate. Both ends of a pair
must be declared in a `.java` file: `implements` is resolved by name, so a
Kotlin class implementing a same-named type is a collision, not an
implementation.

What the Java extractor hands this pass:

- `implements` from classes, enums and records, and the interface's own
  `method` edges. Each overload has its own node with `arity` and `signature`
  metadata (#5251), so overloads pair one-to-one.
- Constructors carry `kind: "constructor"`, so a constructor that shares its
  name with an interface method is never taken for its implementation.

Known limits:

- `extends` is not read, so an abstract-class or subclass override is not a
  target.
- Interface hierarchies (`interface Task extends Worker`) are not walked.
- A `default` method nobody overrides gets no edge: the call really does land
  on the default body.
"""
from __future__ import annotations

from graphify.interface_dispatch import resolve_interface_dispatch

_JAVA_SUFFIXES = (".java",)


def resolve_java_interface_dispatch(
    per_file: list[dict],
    all_nodes: list[dict],
    all_edges: list[dict],
) -> None:
    """Link each Java interface method to every implementer that owns it unambiguously."""
    resolve_interface_dispatch(per_file, all_nodes, all_edges, suffixes=_JAVA_SUFFIXES)
