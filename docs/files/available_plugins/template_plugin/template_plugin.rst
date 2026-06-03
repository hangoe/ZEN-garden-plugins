:orphan:

.. _available_plugins.template_plugin:

Template plugin
----------------

The ``plugin_template`` module shows the most simple implementation of a plugin. It implements
one function that subscribes to the event ``after_model_construction``.
It prints to the console when the event is triggered:

.. literalinclude:: ../../../../zen_garden_plugins/plugin_template/plugin.py
   :language: python

Module documentation
^^^^^^^^^^^^^^^^^^^^

.. automodule:: zen_garden_plugins.plugin_template.plugin
   :members:
   :undoc-members:

