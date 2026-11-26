.. _mcp-server:

Use BiaPy from AI agents (MCP)
==============================

BiaPy includes an `MCP <https://modelcontextprotocol.io>`__ server, so that AI
agents such as Claude Code, Codex, Cursor or Claude Desktop can use BiaPy:
inspect images, create and check configurations, run jobs, follow them and
inspect their results. MCP (Model Context Protocol) is the standard way for
agents to use the tools of other programs.

The server gives the agent the same commands as the :doc:`command-line
interface <cli>`, as tools: each tool runs a ``biapy`` subcommand and returns
the same JSON it prints with ``--json``. It runs on your computer, with your
BiaPy installation, data and GPUs.

Install and add it to your agent
--------------------------------

The installation is explained in :doc:`../get_started/installation` (tab "AI
agents (MCP)"):

.. code-block:: bash

   pip install "biapy[mcp]"
   biapy mcp --setup claude    # or codex; print for any other agent

``biapy mcp --setup`` writes the agent's configuration, so that it starts the
server (``biapy mcp``) by itself in each conversation. Run it in the
environment where BiaPy is installed: the configuration uses the full path of
that environment's ``biapy``, because agents don't have the environment
activated.

* ``--setup claude`` adds it for your user with the ``claude`` command, if it is
  installed. If not (e.g. you use Claude Code only in VS Code), it writes a
  ``.mcp.json`` file in the current folder, which Claude Code uses when that
  folder is opened.
* ``--setup codex`` adds it with the ``codex`` command, if it is installed, or
  else to ``~/.codex/config.toml``.

Then open a new conversation in the agent. In Claude Code, ``/mcp`` shows
``biapy`` as connected.

Other agents, by hand
~~~~~~~~~~~~~~~~~~~~~

``biapy mcp --setup print`` prints the configuration to add it to any other
agent, e.g. Claude Desktop (``claude_desktop_config.json``) or Cursor
(``.cursor/mcp.json``):

.. code-block:: json

   {
     "mcpServers": {
       "biapy": {"command": "/path/to/env/bin/biapy", "args": ["mcp"]}
     }
   }

The path is the output of ``which biapy`` (``where biapy`` on Windows) in the
environment where BiaPy is installed.

Tools
-----

The agent sees these tools, with their descriptions and arguments. Each one is
a ``biapy`` subcommand: see :doc:`cli` for what each returns.

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Tool
     - Subcommand
     - What it does
   * - ``list_workflows``
     - ``list-workflows``
     - The workflows BiaPy can run.
   * - ``inspect_image``
     - ``inspect-image``
     - Describe an image: 2D or 3D, channels.
   * - ``get_config_template``
     - ``get-config-template``
     - The basic configuration of a workflow.
   * - ``create_config``
     - ``create-config``
     - Create a configuration adapted to the data, as the wizard does.
   * - ``validate_config``
     - ``validate-config``
     - Check a configuration.
   * - ``check_data``
     - ``check-data``
     - Check that the data can be used.
   * - ``search_bmz_models``
     - ``search-bmz-models``
     - Search the BioImage Model Zoo for pretrained models BiaPy can use.
   * - ``plan``
     - ``plan``
     - What running a configuration would do, without running it.
   * - ``run``
     - ``run``
     - Run a job in the background.
   * - ``job_status``
     - ``job-status``
     - Whether a job is running, where it is and, if it failed, why.
   * - ``stop_job``
     - ``stop-job``
     - Stop a job started with ``run``.
   * - ``inspect_results``
     - ``inspect-results``
     - What a job left: training, metrics, statistics of the predictions and
       warnings.

The server also gives the agent short instructions on how to use them: the
usual order is to inspect the data, create a configuration, ``plan`` it,
``run`` it, follow it with ``job_status`` and, when it ends, look at
``inspect_results``.

Notes
-----

* Jobs started with ``run`` keep running in the background when the
  conversation ends. Follow them later with ``job_status``.
* Without a GPU, jobs run on the CPU, which is much slower.
* Some tools can take minutes (``create_config`` and ``check_data`` with large
  data, ``search_bmz_models``). If the agent stops waiting, raise its timeout
  for tools.
* Agents with a terminal can also use the ``biapy`` subcommands directly; the
  server makes BiaPy visible and easy to use for them, and usable at all for
  agents without a terminal.

Try it without an agent
-----------------------

The `MCP Inspector <https://github.com/modelcontextprotocol/inspector>`__
(it needs Node.js) starts the server and lets you see its tools and call them:

.. code-block:: bash

   npx @modelcontextprotocol/inspector /path/to/env/bin/biapy mcp

``biapy mcp`` alone waits for an agent to talk to it, through its standard
input, so it shows nothing when run in a terminal.
