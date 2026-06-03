"""
A template for a plugin.

Write functions that subscribe to an event in ZEN-garden. These functions are executed
when ZEN-garden reaches the trigger to the respective event.
"""

from typing import Any

from zen_garden.plugin_system.events import (  # type: ignore[import-untyped]
    Event,
    EventPublisher,
)

# The config can be filled with parameters to be passed to the plugin. Define default
# parameters here. You can pass other values with the config in ZEN-garden.
config: dict[str, Any] = {}


# Choose the event that will trigger the function call
@EventPublisher.register(Event.test_event1)
def function_to_be_called_at_test_event1(*args, **kwargs):
    """This function will be called when the execution reaches the trigger to the event.

    You can implement e.g. new constraints or variables as a plugin which are added
    to the model.
    Make sure the function signature matches with event trigger in ZEN-garden:

    for e.g.:
    ``EventPublisher.trigger(Event.after_model_construction,
    optimization_setup=optimization_setup)``

    the function definition has to be:
    ``def function_to_be_called_at_after_model_construction(optimization_setup):``

    """
    print("Hello. This is the plugin speaking")
