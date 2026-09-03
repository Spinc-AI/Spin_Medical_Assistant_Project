"""Safety layers around the model call.

`precheck` runs on raw user input, `output_validator` on the raw model reply
(structure only), `postcheck` on the parsed response (urgency escalation).
"""
