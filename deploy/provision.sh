#!/usr/bin/env bash
# provision.sh: one-time setup of FAFFAbout on the droplet. Run as root ON THE DROPLET, from
# /opt/faffabout after deploy.sh has copied deploy/ and requirements-server.txt there:
#
#   ssh root@mdeller.com 'bash /opt/faffabout/deploy/provision.sh'
#
# Idempotent: every step checks before it acts, and a second run changes nothing. It
# follows the droplet's existing pattern (one system user per app under /opt/<app>, a
# <app>-web.service running gunicorn on a local port, an nginx site converted to HTTPS by
# certbot) and touches no other app's files.
set -euo pipefail

APP=faffabout
DIR=/opt/$APP
DOMAIN=faffabout.mdeller.com
MMSEQS_RELEASE=18-8cc5c        # must match the version that built data/search/archiveDB.idx
PY=3.14                        # the project's Python; the distro ships 3.12

say() { printf '\n--- %s\n' "$*"; }

say "system user and directory"
id -u "$APP" >/dev/null 2>&1 || useradd --system --home-dir "$DIR" --shell /usr/sbin/nologin "$APP"
mkdir -p "$DIR"/{app,scripts,deploy,baseline/models,data/parquet,data/search}

say "Python $PY via uv"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh
fi
if [[ ! -x "$DIR/.venv/bin/python" ]]; then
  UV_PYTHON_INSTALL_DIR="$DIR/.python" uv venv --python "$PY" "$DIR/.venv"
fi
UV_PYTHON_INSTALL_DIR="$DIR/.python" uv pip install --quiet --python "$DIR/.venv/bin/python" -r "$DIR/requirements-server.txt"
"$DIR/.venv/bin/python" --version

say "MMseqs2 $MMSEQS_RELEASE"
if ! (command -v mmseqs >/dev/null && mmseqs version | grep -q "$MMSEQS_RELEASE"); then
  tmp=$(mktemp -d)
  curl -fsSL -o "$tmp/mmseqs.tar.gz" \
    "https://github.com/soedinglab/MMseqs2/releases/download/$MMSEQS_RELEASE/mmseqs-linux-avx2.tar.gz"
  tar -xzf "$tmp/mmseqs.tar.gz" -C "$tmp"
  install -m 0755 "$tmp/mmseqs/bin/mmseqs" /usr/local/bin/mmseqs
  rm -rf "$tmp"
fi
mmseqs version

say "service environment"
# Two cores are shared with a dozen other apps: do not let one search take all of them.
[[ -f "$DIR/.env" ]] || printf 'FAFFABOUT_MMSEQS_THREADS=2\n' > "$DIR/.env"
chown -R "$APP:$APP" "$DIR"

say "systemd unit"
install -m 0644 "$DIR/deploy/$APP-web.service" "/etc/systemd/system/$APP-web.service"
systemctl daemon-reload
systemctl enable "$APP-web.service"

say "nginx site"
SITE=/etc/nginx/sites-available/$APP
# Never overwrite the site once certbot has rewritten it for HTTPS.
if [[ ! -f "$SITE" ]] || ! grep -q ssl_certificate "$SITE"; then
  install -m 0644 "$DIR/deploy/$APP.nginx.conf" "$SITE"
fi
ln -sf "$SITE" "/etc/nginx/sites-enabled/$APP"
nginx -t
systemctl reload nginx

say "HTTPS certificate"
if [[ ! -d "/etc/letsencrypt/live/$DOMAIN" ]]; then
  certbot --nginx -d "$DOMAIN" --redirect --non-interactive --agree-tos
fi
# certbot writes `listen 443 ssl;`; the other sites here run http2, and on nginx 1.24 it
# belongs on the listen line.
sed -i -E 's/^(\s*listen (\[::\]:)?443 ssl)( ipv6only=on)?;/\1 http2\3;/' "$SITE"
nginx -t
systemctl reload nginx

say "done: ship the code and data with deploy/deploy.sh --go, which starts $APP-web"
