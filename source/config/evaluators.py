#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define the functions to check that :class:`.Evaluator` will initialize."""


IMPLEMENTED_EVALUATORS = (
    "no power loss",
    "transverse eps_x shall not grow too much",
    "transverse eps_y shall not grow too much",
    "longitudinal eps shall not grow too much",
    "max of 99percent transverse eps_x shall not be too high",
    "max of 99percent transverse eps_y shall not be too high",
    "max of 99percent longitudinal eps shall not be too high",
    "longitudinal eps at end",
    "transverse eps at end",
    "mismatch factor at end",
    "transverse mismatch factor at end",
)  #:


def test(beam_calc_post: list[str], **evaluators_kw: str) -> None:
    """Check that desired evaluators are implemented."""
    for evaluator_name in beam_calc_post:
        assert evaluator_name in IMPLEMENTED_EVALUATORS, (
            f"{evaluator_name = } is not implemented."
        )
