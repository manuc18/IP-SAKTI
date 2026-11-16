"""IP-SAKTI prototype application layer.

A thin shell around the existing pipeline. It performs no retrieval logic, no
policy decision and no validation of its own: it builds the components Phases
4-7 already provide, calls them in the order those phases established, and
serialises the result. Every safety property — citation validation, grounding,
safe abstention — belongs to `ai/answering` and is not restated here.
"""
