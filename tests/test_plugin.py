import importlib
import sys
import types


def _load_plugin_with_fake_events(monkeypatch):
    """
    Import plugin module with a fake zen_garden events module.

    Behaves like the Zen garden plugin loader and returns all modules, events and calls loaded.
    """
    calls = []

    class Event:
        test_event1 = object()

    class EventPublisher:
        @staticmethod
        def register(event):
            def decorator(func):
                calls.append((event, func))
                return func

            return decorator

    zen_garden = types.ModuleType("zen_garden")
    plugin_system = types.ModuleType("zen_garden.plugin_system")
    events = types.ModuleType("zen_garden.plugin_system.events")
    events.Event = Event
    events.EventPublisher = EventPublisher

    # Build the module chain expected by plugin_template.plugin imports.
    zen_garden.plugin_system = plugin_system
    plugin_system.events = events

    monkeypatch.setitem(sys.modules, "zen_garden", zen_garden)
    monkeypatch.setitem(sys.modules, "zen_garden.plugin_system", plugin_system)
    monkeypatch.setitem(sys.modules, "zen_garden.plugin_system.events", events)
    monkeypatch.delitem(sys.modules, "plugin_template.plugin", raising=False)

    module = importlib.import_module("plugin_template.plugin")
    return module, Event, calls


def test_plugin_exposes_config_dict(monkeypatch):
    """Test config exposed by plugin."""
    module, _event, _calls = _load_plugin_with_fake_events(monkeypatch)

    assert isinstance(module.config, dict)


def test_plugin_registers_handler_for_test_event1(monkeypatch):
    """Test handler registered for test event1."""
    module, event, calls = _load_plugin_with_fake_events(monkeypatch)

    assert len(calls) == 1
    registered_event, registered_function = calls[0]
    assert registered_event is event.test_event1
    assert registered_function is module.this_will_be_called_first

