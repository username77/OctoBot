# -*- mode: python -*-
from PyInstaller.utils.hooks import collect_submodules

block_cipher = None

OCTOBOT_PACKAGES_FILES = REQUIRED = [s.strip() for s in open('bin/octobot_packages_files.txt').readlines()]

LOCAL_PACKAGES_SUBMODULES = (
    collect_submodules('octobot_commons') +
    collect_submodules('octobot_trading') +
    collect_submodules('octobot_backtesting') +
    collect_submodules('octobot_evaluators') +
    collect_submodules('octobot_services') +
    collect_submodules('octobot_tentacles_manager') +
    collect_submodules('octobot_node') +
    collect_submodules('async_channel') +
    collect_submodules('octobot_flow') +
    collect_submodules('octobot_sync') +
    collect_submodules('trading_backend')
)

# hiddenimports=['numpy.core._dtype_ctypes'] from https://github.com/pyinstaller/pyinstaller/issues/3982
a = Analysis(
   ['../start.py'],
   pathex=[
      '../',
      '../packages/agents',
      '../packages/async_channel',
      '../packages/backtesting',
      '../packages/commons',
      '../packages/evaluators',
      '../packages/flow',
      '../packages/node',
      '../packages/services',
      '../packages/sync',
      '../packages/tentacles_manager',
      '../packages/trading',
      '../packages/trading_backend',
   ],
   datas=[
      ('../octobot/config', 'octobot/config'),
      ('../octobot/strategy_optimizer/optimizer_data_files', 'octobot/strategy_optimizer/optimizer_data_files')
   ],
   hiddenimports=[
      "colorlog", "numpy.core._dtype_ctypes", "dotenv",
      "pgpy", "imghdr",
      "aiosqlite", "aiohttp",
      "pyarrow", "pyiceberg",
      "psutil",
      "telegram", "telegram.ext", "telethon", "jsonschema",
      "tulipy",
      "asyncpraw", "simplifiedpytrends", "simplifiedpytrends.exceptions", "simplifiedpytrends.request",
      "pyngrok", "pyngrok.ngrok", "openai",
      "flask", "flask_login", "flask_wtf", "flask_caching", "flask_compress", "flask_socketio", "flask_cors",
      "wtforms", "wtforms.fields", "gevent", "geventwebsocket",
      "vaderSentiment", "vaderSentiment.vaderSentiment",
      "coingecko_openapi_client",
      "certifi",
      "aiofiles",
      "pydantic", "mcp",
      "dbos", "fastapi", "passlib", "fastapi.staticfiles",
      "web3",
      "ccxt", "ccxt.async_support", "ccxt.pro", "order_book", "cmath", "cryptography", "websockets", "yarl", "idna", "sortedcontainers",
      "websockets.legacy", "websockets.legacy.auth", "websockets.legacy.client", "websockets.legacy.compatibility",
      "websockets.legacy.framing", "websockets.legacy.handshake", "websockets.legacy.http", "websockets.legacy.protocol",
      "websockets.legacy.server",
      "engineio.async_drivers.gevent"
   ] + OCTOBOT_PACKAGES_FILES + LOCAL_PACKAGES_SUBMODULES,
   excludes=["tentacles", "logs", "user"],
   hookspath=[],
   runtime_hooks=[],
   win_no_prefer_redirects=False,
   win_private_assemblies=False,
   cipher=block_cipher
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name='OctoBot',
          debug=False,
          strip=False,
          icon="favicon.ico",
          upx=True,
          runtime_tmpdir=None,
          console=True )
