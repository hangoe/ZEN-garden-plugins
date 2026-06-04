.. _installation.installation:

###########################
Installation for Developers
###########################

If you want to work on the codebase, you can fork and clone the repository from
`GitHub <https://github.com/ZEN-universe/ZEN-garden-plugins>`_.

If it's your first time using GitHub, register at `<https://github.com/>`_.
After you have created an account, you can fork and clone the repository.

Navigate to `<https://github.com/ZEN-universe/ZEN-garden-plugins>`_ on Github and click
on the "Fork" button at the top right corner of the page to create a copy of the
repository under your account and select yourself as the owner.

.. image:: ../figures/quick_start/create_fork.png
    :alt: creating a fork

|

**Clone your forked repository:**

Clone your forked repository by running the following lines in `Git-Bash
<https://git-scm.com/downloads>`_::

    git clone git@github.com:<your-username>/ZEN-garden-plugins.git
    cd ZEN-garden-plugins

Substitute ``<your-username>`` with your Github username. If you gave the forked
repository a different name, replace ``ZEN-garden-plugins`` with the name of your
repository.

.. note::
    If you get the permissions error "Permission denied (publickey)", you will
    need to create the SSH key. Follow the instructions on `how to generate an
    SSH key <https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent#generating-a-new-ssh-key>`_
    and then `how to add it to your account <https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account#adding-a-new-ssh-key-to-your-account>`_.
    You will not need to add the SSH key to the Agent, so only follow the first
    website until before `Adding your SSH key to the ssh-agent <https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent#adding-your-ssh-key-to-the-ssh-agent>`_

**Track the upstream repository:**

In your terminal window, navigate to the folder in which ZEN-garden was
installed (i.e. the folder where the file ``zen_garden_plugins_env.yml`` is located)::

    cd <path_to_zen_garden_repo>

Track the upstream repository by running the following lines in Git-Bash::

    git remote add upstream git@github.com:ZEN-universe/ZEN-garden-plugins.git
    git fetch upstream

**Create the ZEN-garden-plugins conda environment:**

Open the Anaconda Prompt application. This is a terminal window provided by
Anaconda which allows you to run Anaconda commands.

In the Anaconda Prompt, change the directory to the root directory of your
local ZEN-garden repository i.e. the folder where the file
``zen_garden_env.yml`` is located::

  cd <path_to_zen_garden_plugins_repo>

Now you can install the conda environment for zen-garden with the following
command::

  conda env create -f zen_garden_plugins_env.yml

The installation may take a couple of minutes. If the installation was
successful, you can see the environment at
``C:\Users\<username>\anaconda3\envs`` or wherever Anaconda is installed.

In the new environment, setup the pre-commit hooks by running the following command in the Anaconda Prompt::

  pre-commit install

These pre-commit hooks will automatically format and lint your code when you commit 
changes to the repository. This ensures that the codebase remains clean and 
consistent.

.. note::
    If you forked the ZEN-garden repository and created the environment from
    ``zen_garden_plugins_env.yml``, then the environment will by default be
    called ``zen-garden-plugins-env``.

.. note::
    We strongly recommend working with conda environments. When installing the
    zen-garden-plugins conda environment via the ``zen_garden_plugins_env.yml``, 
    the zen-garden-plugins package, as well as all other dependencies, are installed automatically.

