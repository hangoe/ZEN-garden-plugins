.. _implementation.implementing_plugins:

####################################
Implementing your own plugin
####################################

This page walks you through creating a plugin for ZEN-garden from scratch.

There are two ways of working with plugins:

- A plugin can live in this repository as a submodule. These plugins must 
  be fully documented, have tests, and be deemed important enough for 
  maintenance by the maintainers.

- You can also create your own plugin in a separate repository and install it as a regular Python package.
  Therefore, you can fork this repository and adapt the plugin template.

In both cases the module name is the name of the plugin that needs to be added to the ZEN-garden
configuration file (``config.json``) to activate the plugin. If you are developing a plugin, you can
install this package in editable mode with ``pip install -e`` to see changes immediately without reinstalling.

.. note::

   Plugins are intentionally kept separate from ZEN-garden. You should never need
   to modify ZEN-garden itself to add a plugin, unless you need a new event that does not
   exist yet.


Step 1 — Create a repository
-------------------------------

Fork the ZEN-garden plugin template `repository on GitHub
<https://github.com/ZEN-universe/ZEN-garden-plugins>`_. You can copy-paste the ``plugin_template/`` directory,
rename it and start developing your own plugin.

All plugin logic goes into ``plugin.py``. You can define helper modules if you like,
but the loader only looks for ``plugin.py``.


Step 2 — Adapt ``pyproject.toml``
----------------------------------

The plugin must be a valid Python package with a ``pyproject.toml`` that advertises
itself as a plugin for ZEN-garden. You can adapt the template's ``pyproject.toml``. Essentially you only need
to change the name. In case your plugin relies on additional packages, you also need to specify the
dependencies.

Step 3 — Develop your plugin
-----------------------------

Every ``plugin.py`` must contain:

1. A ``config`` dictionary at the top level.  ZEN-garden will fill this with the
   settings you specify in ``config.json`` under the respective plugin name.
2. One or more functions decorated with
   ``@EventPublisher.register(Event.<event_name>)``.  These functions are called
   automatically when ZEN-garden reaches the corresponding event.
3. Testing: create a new directory ``tests/`` and add test files there.
4. Documentation: create a new directory under ``docs/files/available_plugins`` and add your
   documentation there. You can adapt the template's documentation.

Install the plugin package into the same Python environment as ZEN-garden:

.. code-block:: shell

    pip install -e path/to/my_plugin

The ``-e`` flag installs it in *editable* mode, which means changes to your files
take effect immediately without reinstalling.


Available events
~~~~~~~~~~~~~~~~

Events are defined in ``zen_garden.plugin_system.events.Event``.
Each event corresponds to a specific point in the workflow and passes relevant
objects as keyword arguments. Refer to the ZEN-garden documentation or source code
for the list of available events and their arguments.

Step 5 — Activate in ``config.json``
--------------------------------------

Add the plugin's name to the ZEN-garden ``config.json``:

.. code-block:: json

    {
        "plugins": {
            "my_plugin_name": {
                "my_setting": 123
            }
        }
    }

The dictionary under ``"my_plugin_name"`` is merged into the plugin's ``config``
before any of its functions are called.  Use it to pass numerical parameters,
file paths, flags, and so on.

You can see the template plugin's documentation here: :ref:`available_plugins.template_plugin`.

How the loader works
---------------------

When ZEN-garden starts, the loader scans all installed packages for entry points
in the ``zen_garden.plugins`` group.  For each plugin listed in ``config.json`` it:

1. Locates the installed entry point with that name.
2. Imports the corresponding ``plugin.py`` module.
3. Merges the user-provided settings into the module's ``config`` dictionary.

If a plugin name appears in ``config.json`` but is not installed, ZEN-garden raises
a ``ModuleNotFoundError`` with a clear message listing the available plugins.
