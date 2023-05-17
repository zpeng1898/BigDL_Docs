- DGL Node Classification Distributed Training
[Docs](https://docs.dgl.ai/en/latest/tutorials/dist/1_node_classification.html#set-up-distributed-training-environment)

- Environment Preparation
Prepare 2 nodes `172.16.0.107` and `172.16.0.136`.

conda environment:
```
conda create -n py376 python=3.7
pip install torch torchvision torchmetrics==0.10.0 tqdm 
pip install --pre --upgrade bigdl-orca[ray]
pip install --pre dgl -f https://data.dgl.ai/wheels-test/repo.html
pip install --pre dglgo -f https://data.dgl.ai/wheels-test/repo.html
python -c "import ogb; print(ogb.__version__)"
pip install -U ogb
```

- python scripts
```
python3 partition_data.py
```
```
python3 launch.py               \
  --workspace /home/kai/pzy/0419/1          \
  --num_trainers 4                  \
  --num_samplers 1                    \
  --num_servers 1                     \
  --part_config 4part_data/ogbn-arxiv.json     \
  --ip_config ip_config.txt           \
  --ssh_port 22                    \
  "/home/kai/anaconda3/envs/py376/bin/python3 dgl1.py"  \
  --extra_envs MASTER_ADDR=172.16.0.107 MASTER_PORT=1234
```

- Results

train 1 epoch getting `Epoch 0: Validation Accuracy 0.44414607948442536`.
```
(base) [root@Almaren-Node-107 1]# python3 launch.py                 --workspace /home/kai/pzy/0419/1            --num_trainers 4                    --num_samplers 1                      --num_servers 1                       --part_config 4part_data/ogbn-arxiv.json       --ip_config ip_config.txt             --ssh_port 22                      "/home/kai/anaconda3/envs/py376/bin/python3 dgl1.py"    --extra_envs MASTER_ADDR=172.16.0.107 MASTER_PORT=1234
The number of OMP threads per trainer is set to 11
OpenSSH_7.4p1, OpenSSL 1.0.2k-fips  26 Jan 2017
debug1: Reading configuration data /etc/ssh/ssh_config
debug1: /etc/ssh/ssh_config line 58: Applying options for *
debug1: Connecting to 172.16.0.107 [172.16.0.107] port 22.
debug1: Connection established.
debug1: permanently_set_uid: 0/0
debug1: identity file /root/.ssh/id_rsa type 1
debug1: key_load_public: No such file or directory
debug1: identity file /root/.ssh/id_rsa-cert type -1
debug1: key_load_public: No such file or directory
debug1: identity file /root/.ssh/id_dsa type -1
debug1: key_load_public: No such file or directory
debug1: identity file /root/.ssh/id_dsa-cert type -1
debug1: key_load_public: No such file or directory
debug1: identity file /root/.ssh/id_ecdsa type -1
debug1: key_load_public: No such file or directory
debug1: identity file /root/.ssh/id_ecdsa-cert type -1
debug1: key_load_public: No such file or directory
debug1: identity file /root/.ssh/id_ed25519 type -1
debug1: key_load_public: No such file or directory
debug1: identity file /root/.ssh/id_ed25519-cert type -1
debug1: Enabling compatibility mode for protocol 2.0
debug1: Local version string SSH-2.0-OpenSSH_7.4
debug1: Remote protocol version 2.0, remote software version OpenSSH_7.4
debug1: match: OpenSSH_7.4 pat OpenSSH* compat 0x04000000
debug1: Authenticating to 172.16.0.107:22 as 'root'
debug1: SSH2_MSG_KEXINIT sent
debug1: SSH2_MSG_KEXINIT received
debug1: kex: algorithm: curve25519-sha256
debug1: kex: host key algorithm: ecdsa-sha2-nistp256
debug1: kex: server->client cipher: chacha20-poly1305@openssh.com MAC: <implicit> compression: none
debug1: kex: client->server cipher: chacha20-poly1305@openssh.com MAC: <implicit> compression: none
debug1: kex: curve25519-sha256 need=64 dh_need=64
debug1: kex: curve25519-sha256 need=64 dh_need=64
debug1: expecting SSH2_MSG_KEX_ECDH_REPLY
debug1: Server host key: ecdsa-sha2-nistp256 SHA256:eNjOom+lPkILffTFqXWrobyQTEn4Wp2qAt4TAPUBALc
debug1: Host '172.16.0.107' is known and matches the ECDSA host key.
debug1: Found key in /root/.ssh/known_hosts:25
debug1: rekey after 134217728 blocks
debug1: SSH2_MSG_NEWKEYS sent
debug1: expecting SSH2_MSG_NEWKEYS
debug1: SSH2_MSG_NEWKEYS received
debug1: rekey after 134217728 blocks
debug1: SSH2_MSG_EXT_INFO received
debug1: kex_input_ext_info: server-sig-algs=<rsa-sha2-256,rsa-sha2-512>
debug1: SSH2_MSG_SERVICE_ACCEPT received
debug1: Authentications that can continue: publickey,gssapi-keyex,gssapi-with-mic,password
debug1: Next authentication method: gssapi-keyex
debug1: No valid Key exchange context
debug1: Next authentication method: gssapi-with-mic
debug1: Unspecified GSS failure.  Minor code may provide more information
No Kerberos credentials available (default cache: KEYRING:persistent:0)

debug1: Unspecified GSS failure.  Minor code may provide more information
No Kerberos credentials available (default cache: KEYRING:persistent:0)

debug1: Next authentication method: publickey
debug1: Offering RSA public key: /root/.ssh/id_rsa
debug1: Server accepts key: pkalg rsa-sha2-512 blen 535
debug1: Authentication succeeded (publickey).
Authenticated to 172.16.0.107 ([172.16.0.107]:22).
debug1: channel 0: new [client-session]
debug1: Requesting no-more-sessions@openssh.com
debug1: Entering interactive session.
debug1: pledge: network
debug1: client_input_global_request: rtype hostkeys-00@openssh.com want_reply 0
debug1: Sending environment.
debug1: Sending env LANG = en_US.UTF-8
debug1: Sending command: cd /home/kai/pzy/0419/1; (export MASTER_ADDR=172.16.0.107 MASTER_PORT=1234; (export DGL_ROLE=server DGL_NUM_SAMPLER=1 OMP_NUM_THREADS=1 DGL_NUM_CLIENT=16 DGL_CONF_PATH=4part_data/ogbn-arxiv.json DGL_IP_CONFIG=ip_config.txt DGL_NUM_SERVER=1 DGL_GRAPH_FORMAT=csc DGL_KEEP_ALIVE=0  DGL_SERVER_ID=0; /home/kai/anaconda3/envs/py376/bin/python3 dgl1.py))
OpenSSH_7.4p1, OpenSSL 1.0.2k-fips  26 Jan 2017
debug1: Reading configuration data /etc/ssh/ssh_config
debug1: /etc/ssh/ssh_config line 58: Applying options for *
debug1: Connecting to 172.16.0.136 [172.16.0.136] port 22.
debug1: Connection established.
debug1: permanently_set_uid: 0/0
debug1: identity file /root/.ssh/id_rsa type 1
debug1: key_load_public: No such file or directory
debug1: identity file /root/.ssh/id_rsa-cert type -1
debug1: key_load_public: No such file or directory
debug1: identity file /root/.ssh/id_dsa type -1
debug1: key_load_public: No such file or directory
debug1: identity file /root/.ssh/id_dsa-cert type -1
debug1: key_load_public: No such file or directory
debug1: identity file /root/.ssh/id_ecdsa type -1
debug1: key_load_public: No such file or directory
debug1: identity file /root/.ssh/id_ecdsa-cert type -1
debug1: key_load_public: No such file or directory
debug1: identity file /root/.ssh/id_ed25519 type -1
debug1: key_load_public: No such file or directory
debug1: identity file /root/.ssh/id_ed25519-cert type -1
debug1: Enabling compatibility mode for protocol 2.0
debug1: Local version string SSH-2.0-OpenSSH_7.4
debug1: Remote protocol version 2.0, remote software version OpenSSH_7.4
debug1: match: OpenSSH_7.4 pat OpenSSH* compat 0x04000000
debug1: Authenticating to 172.16.0.136:22 as 'root'
debug1: SSH2_MSG_KEXINIT sent
debug1: SSH2_MSG_KEXINIT received
debug1: kex: algorithm: curve25519-sha256
debug1: kex: host key algorithm: ecdsa-sha2-nistp256
debug1: kex: server->client cipher: chacha20-poly1305@openssh.com MAC: <implicit> compression: none
debug1: kex: client->server cipher: chacha20-poly1305@openssh.com MAC: <implicit> compression: none
debug1: kex: curve25519-sha256 need=64 dh_need=64
debug1: kex: curve25519-sha256 need=64 dh_need=64
debug1: expecting SSH2_MSG_KEX_ECDH_REPLY
debug1: Server host key: ecdsa-sha2-nistp256 SHA256:zJaTY8T35Tf9ZktxdHU/foeYP87c4RPizp21o/ansnk
debug1: Host '172.16.0.136' is known and matches the ECDSA host key.
debug1: Found key in /root/.ssh/known_hosts:40
debug1: rekey after 134217728 blocks
debug1: SSH2_MSG_NEWKEYS sent
debug1: expecting SSH2_MSG_NEWKEYS
debug1: SSH2_MSG_NEWKEYS received
debug1: rekey after 134217728 blocks
debug1: SSH2_MSG_EXT_INFO received
debug1: kex_input_ext_info: server-sig-algs=<rsa-sha2-256,rsa-sha2-512>
debug1: SSH2_MSG_SERVICE_ACCEPT received
debug1: Authentications that can continue: publickey,gssapi-keyex,gssapi-with-mic,password
debug1: Next authentication method: gssapi-keyex
debug1: No valid Key exchange context
debug1: Next authentication method: gssapi-with-mic
debug1: Unspecified GSS failure.  Minor code may provide more information
No Kerberos credentials available (default cache: KEYRING:persistent:0)

debug1: Unspecified GSS failure.  Minor code may provide more information
No Kerberos credentials available (default cache: KEYRING:persistent:0)

debug1: Next authentication method: publickey
debug1: Offering RSA public key: /root/.ssh/id_rsa
debug1: Server accepts key: pkalg rsa-sha2-512 blen 535
debug1: Authentication succeeded (publickey).
Authenticated to 172.16.0.136 ([172.16.0.136]:22).
debug1: channel 0: new [client-session]
debug1: Requesting no-more-sessions@openssh.com
debug1: Entering interactive session.
debug1: pledge: network
debug1: client_input_global_request: rtype hostkeys-00@openssh.com want_reply 0
debug1: Sending environment.
debug1: Sending env LANG = en_US.UTF-8
debug1: Sending command: cd /home/kai/pzy/0419/1; (export MASTER_ADDR=172.16.0.107 MASTER_PORT=1234; (export DGL_ROLE=server DGL_NUM_SAMPLER=1 OMP_NUM_THREADS=1 DGL_NUM_CLIENT=16 DGL_CONF_PATH=4part_data/ogbn-arxiv.json DGL_IP_CONFIG=ip_config.txt DGL_NUM_SERVER=1 DGL_GRAPH_FORMAT=csc DGL_KEEP_ALIVE=0  DGL_SERVER_ID=1; /home/kai/anaconda3/envs/py376/bin/python3 dgl1.py))
OpenSSH_7.4p1, OpenSSL 1.0.2k-fips  26 Jan 2017
debug1: Reading configuration data /etc/ssh/ssh_config
debug1: /etc/ssh/ssh_config line 58: Applying options for *
debug1: Connecting to 172.16.0.107 [172.16.0.107] port 22.
debug1: Connection established.
debug1: permanently_set_uid: 0/0
debug1: identity file /root/.ssh/id_rsa type 1
debug1: key_load_public: No such file or directory
debug1: identity file /root/.ssh/id_rsa-cert type -1
debug1: key_load_public: No such file or directory
debug1: identity file /root/.ssh/id_dsa type -1
debug1: key_load_public: No such file or directory
debug1: identity file /root/.ssh/id_dsa-cert type -1
debug1: key_load_public: No such file or directory
debug1: identity file /root/.ssh/id_ecdsa type -1
debug1: key_load_public: No such file or directory
debug1: identity file /root/.ssh/id_ecdsa-cert type -1
debug1: key_load_public: No such file or directory
debug1: identity file /root/.ssh/id_ed25519 type -1
debug1: key_load_public: No such file or directory
debug1: identity file /root/.ssh/id_ed25519-cert type -1
debug1: Enabling compatibility mode for protocol 2.0
debug1: Local version string SSH-2.0-OpenSSH_7.4
debug1: Remote protocol version 2.0, remote software version OpenSSH_7.4
debug1: match: OpenSSH_7.4 pat OpenSSH* compat 0x04000000
debug1: Authenticating to 172.16.0.107:22 as 'root'
debug1: SSH2_MSG_KEXINIT sent
debug1: SSH2_MSG_KEXINIT received
debug1: kex: algorithm: curve25519-sha256
debug1: kex: host key algorithm: ecdsa-sha2-nistp256
debug1: kex: server->client cipher: chacha20-poly1305@openssh.com MAC: <implicit> compression: none
debug1: kex: client->server cipher: chacha20-poly1305@openssh.com MAC: <implicit> compression: none
debug1: kex: curve25519-sha256 need=64 dh_need=64
debug1: kex: curve25519-sha256 need=64 dh_need=64
debug1: expecting SSH2_MSG_KEX_ECDH_REPLY
debug1: Server host key: ecdsa-sha2-nistp256 SHA256:eNjOom+lPkILffTFqXWrobyQTEn4Wp2qAt4TAPUBALc
debug1: Host '172.16.0.107' is known and matches the ECDSA host key.
debug1: Found key in /root/.ssh/known_hosts:25
debug1: rekey after 134217728 blocks
debug1: SSH2_MSG_NEWKEYS sent
debug1: expecting SSH2_MSG_NEWKEYS
debug1: SSH2_MSG_NEWKEYS received
debug1: rekey after 134217728 blocks
debug1: SSH2_MSG_EXT_INFO received
debug1: kex_input_ext_info: server-sig-algs=<rsa-sha2-256,rsa-sha2-512>
debug1: SSH2_MSG_SERVICE_ACCEPT received
debug1: Authentications that can continue: publickey,gssapi-keyex,gssapi-with-mic,password
debug1: Next authentication method: gssapi-keyex
debug1: No valid Key exchange context
debug1: Next authentication method: gssapi-with-mic
debug1: Unspecified GSS failure.  Minor code may provide more information
No Kerberos credentials available (default cache: KEYRING:persistent:0)

debug1: Unspecified GSS failure.  Minor code may provide more information
No Kerberos credentials available (default cache: KEYRING:persistent:0)

debug1: Next authentication method: publickey
debug1: Offering RSA public key: /root/.ssh/id_rsa
debug1: Server accepts key: pkalg rsa-sha2-512 blen 535
debug1: Authentication succeeded (publickey).
Authenticated to 172.16.0.107 ([172.16.0.107]:22).
debug1: channel 0: new [client-session]
debug1: Requesting no-more-sessions@openssh.com
debug1: Entering interactive session.
debug1: pledge: network
debug1: client_input_global_request: rtype hostkeys-00@openssh.com want_reply 0
debug1: Sending environment.
debug1: Sending env LANG = en_US.UTF-8
debug1: Sending command: cd /home/kai/pzy/0419/1; (export MASTER_ADDR=172.16.0.107 MASTER_PORT=1234; (export DGL_DIST_MODE=distributed DGL_ROLE=client DGL_NUM_SAMPLER=1 DGL_NUM_CLIENT=16 DGL_CONF_PATH=4part_data/ogbn-arxiv.json DGL_IP_CONFIG=ip_config.txt DGL_NUM_SERVER=1 DGL_GRAPH_FORMAT=csc OMP_NUM_THREADS=11 DGL_GROUP_ID=0 ; /home/kai/anaconda3/envs/py376/bin/python3 -m torch.distributed.launch --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr=172.16.0.107 --master_port=1234 dgl1.py))
OpenSSH_7.4p1, OpenSSL 1.0.2k-fips  26 Jan 2017
debug1: Reading configuration data /etc/ssh/ssh_config
debug1: /etc/ssh/ssh_config line 58: Applying options for *
debug1: Connecting to 172.16.0.136 [172.16.0.136] port 22.
debug1: Connection established.
debug1: permanently_set_uid: 0/0
debug1: identity file /root/.ssh/id_rsa type 1
debug1: key_load_public: No such file or directory
debug1: identity file /root/.ssh/id_rsa-cert type -1
debug1: key_load_public: No such file or directory
debug1: identity file /root/.ssh/id_dsa type -1
debug1: key_load_public: No such file or directory
debug1: identity file /root/.ssh/id_dsa-cert type -1
debug1: key_load_public: No such file or directory
debug1: identity file /root/.ssh/id_ecdsa type -1
debug1: key_load_public: No such file or directory
debug1: identity file /root/.ssh/id_ecdsa-cert type -1
debug1: key_load_public: No such file or directory
debug1: identity file /root/.ssh/id_ed25519 type -1
debug1: key_load_public: No such file or directory
debug1: identity file /root/.ssh/id_ed25519-cert type -1
debug1: Enabling compatibility mode for protocol 2.0
debug1: Local version string SSH-2.0-OpenSSH_7.4
debug1: Remote protocol version 2.0, remote software version OpenSSH_7.4
debug1: match: OpenSSH_7.4 pat OpenSSH* compat 0x04000000
debug1: Authenticating to 172.16.0.136:22 as 'root'
debug1: SSH2_MSG_KEXINIT sent
debug1: SSH2_MSG_KEXINIT received
debug1: kex: algorithm: curve25519-sha256
debug1: kex: host key algorithm: ecdsa-sha2-nistp256
debug1: kex: server->client cipher: chacha20-poly1305@openssh.com MAC: <implicit> compression: none
debug1: kex: client->server cipher: chacha20-poly1305@openssh.com MAC: <implicit> compression: none
debug1: kex: curve25519-sha256 need=64 dh_need=64
debug1: kex: curve25519-sha256 need=64 dh_need=64
debug1: expecting SSH2_MSG_KEX_ECDH_REPLY
debug1: Server host key: ecdsa-sha2-nistp256 SHA256:zJaTY8T35Tf9ZktxdHU/foeYP87c4RPizp21o/ansnk
debug1: Host '172.16.0.136' is known and matches the ECDSA host key.
debug1: Found key in /root/.ssh/known_hosts:40
debug1: rekey after 134217728 blocks
debug1: SSH2_MSG_NEWKEYS sent
debug1: expecting SSH2_MSG_NEWKEYS
debug1: SSH2_MSG_NEWKEYS received
debug1: rekey after 134217728 blocks
debug1: SSH2_MSG_EXT_INFO received
debug1: kex_input_ext_info: server-sig-algs=<rsa-sha2-256,rsa-sha2-512>
debug1: SSH2_MSG_SERVICE_ACCEPT received
debug1: Authentications that can continue: publickey,gssapi-keyex,gssapi-with-mic,password
debug1: Next authentication method: gssapi-keyex
debug1: No valid Key exchange context
debug1: Next authentication method: gssapi-with-mic
debug1: Unspecified GSS failure.  Minor code may provide more information
No Kerberos credentials available (default cache: KEYRING:persistent:0)

debug1: Unspecified GSS failure.  Minor code may provide more information
No Kerberos credentials available (default cache: KEYRING:persistent:0)

debug1: Next authentication method: publickey
debug1: Offering RSA public key: /root/.ssh/id_rsa
debug1: Server accepts key: pkalg rsa-sha2-512 blen 535
debug1: Authentication succeeded (publickey).
Authenticated to 172.16.0.136 ([172.16.0.136]:22).
debug1: channel 0: new [client-session]
debug1: Requesting no-more-sessions@openssh.com
debug1: Entering interactive session.
debug1: pledge: network
debug1: client_input_global_request: rtype hostkeys-00@openssh.com want_reply 0
debug1: Sending environment.
debug1: Sending env LANG = en_US.UTF-8
debug1: Sending command: cd /home/kai/pzy/0419/1; (export MASTER_ADDR=172.16.0.107 MASTER_PORT=1234; (export DGL_DIST_MODE=distributed DGL_ROLE=client DGL_NUM_SAMPLER=1 DGL_NUM_CLIENT=16 DGL_CONF_PATH=4part_data/ogbn-arxiv.json DGL_IP_CONFIG=ip_config.txt DGL_NUM_SERVER=1 DGL_GRAPH_FORMAT=csc OMP_NUM_THREADS=11 DGL_GROUP_ID=0 ; /home/kai/anaconda3/envs/py376/bin/python3 -m torch.distributed.launch --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr=172.16.0.107 --master_port=1234 dgl1.py))
cleanupu process runs
[07:37:49] /opt/dgl/src/rpc/rpc.cc:141: Sender with NetType~socket is created.
[07:37:49] /opt/dgl/src/rpc/rpc.cc:161: Receiver with NetType~socket is created.
[07:37:50] /opt/dgl/src/rpc/rpc.cc:141: Sender with NetType~socket is created.
[07:37:50] /opt/dgl/src/rpc/rpc.cc:161: Receiver with NetType~socket is created.
Warning! Interface: eno1 
IP address not available for interface.
Warning! Interface: enp4s0f3 
IP address not available for interface.
Warning! Interface: eno4 
IP address not available for interface.
[07:37:52] /opt/dgl/src/rpc/rpc.cc:141: Sender with NetType~socket is created.
[07:37:52] /opt/dgl/src/rpc/rpc.cc:161: Receiver with NetType~socket is created.
Warning! Interface: eno1 
IP address not available for interface.
Warning! Interface: enp4s0f3 
IP address not available for interface.
Warning! Interface: eno4 
IP address not available for interface.
[07:37:52] /opt/dgl/src/rpc/rpc.cc:141: Sender with NetType~socket is created.
[07:37:52] /opt/dgl/src/rpc/rpc.cc:161: Receiver with NetType~socket is created.
Warning! Interface: eno1 
IP address not available for interface.
Warning! Interface: enp4s0f3 
IP address not available for interface.
Warning! Interface: eno4 
IP address not available for interface.
[07:37:52] /opt/dgl/src/rpc/rpc.cc:141: Sender with NetType~socket is created.
[07:37:52] /opt/dgl/src/rpc/rpc.cc:161: Receiver with NetType~socket is created.
Warning! Interface: eno1 
IP address not available for interface.
Warning! Interface: enp4s0f3 
IP address not available for interface.
Warning! Interface: eno4 
IP address not available for interface.
[07:37:52] /opt/dgl/src/rpc/rpc.cc:141: Sender with NetType~socket is created.
[07:37:52] /opt/dgl/src/rpc/rpc.cc:161: Receiver with NetType~socket is created.
Warning! Interface: eno1 
IP address not available for interface.
Warning! Interface: enp4s0f3 
IP address not available for interface.
Warning! Interface: eno4 
IP address not available for interface.
[07:37:52] /opt/dgl/src/rpc/rpc.cc:141: Sender with NetType~socket is created.
[07:37:52] /opt/dgl/src/rpc/rpc.cc:161: Receiver with NetType~socket is created.
Warning! Interface: eno1 
IP address not available for interface.
Warning! Interface: enp4s0f3 
IP address not available for interface.
Warning! Interface: eno4 
IP address not available for interface.
[07:37:52] /opt/dgl/src/rpc/rpc.cc:141: Sender with NetType~socket is created.
[07:37:52] /opt/dgl/src/rpc/rpc.cc:161: Receiver with NetType~socket is created.
Warning! Interface: eno1 
IP address not available for interface.
Warning! Interface: enp4s0f3 
IP address not available for interface.
Warning! Interface: eno4 
IP address not available for interface.
[07:37:52] /opt/dgl/src/rpc/rpc.cc:141: Sender with NetType~socket is created.
[07:37:52] /opt/dgl/src/rpc/rpc.cc:161: Receiver with NetType~socket is created.
Warning! Interface: eno1 
IP address not available for interface.
Warning! Interface: enp4s0f3 
IP address not available for interface.
Warning! Interface: eno4 
IP address not available for interface.
[07:37:52] /opt/dgl/src/rpc/rpc.cc:141: Sender with NetType~socket is created.
[07:37:52] /opt/dgl/src/rpc/rpc.cc:161: Receiver with NetType~socket is created.
Warning! Interface: eno1 
IP address not available for interface.
Warning! Interface: enp4s0f3 
IP address not available for interface.
Warning! Interface: eno4 
IP address not available for interface.
[07:37:53] /opt/dgl/src/rpc/rpc.cc:141: Sender with NetType~socket is created.
[07:37:53] /opt/dgl/src/rpc/rpc.cc:161: Receiver with NetType~socket is created.
Warning! Interface: eno1 
IP address not available for interface.
Warning! Interface: enp4s0f3 
IP address not available for interface.
Warning! Interface: eno4 
IP address not available for interface.
[07:37:54] /opt/dgl/src/rpc/rpc.cc:141: Sender with NetType~socket is created.
[07:37:54] /opt/dgl/src/rpc/rpc.cc:161: Receiver with NetType~socket is created.
Warning! Interface: eno1 
IP address not available for interface.
Warning! Interface: enp4s0f3 
IP address not available for interface.
Warning! Interface: eno4 
IP address not available for interface.
[07:37:54] /opt/dgl/src/rpc/rpc.cc:141: Sender with NetType~socket is created.
[07:37:54] /opt/dgl/src/rpc/rpc.cc:161: Receiver with NetType~socket is created.
Warning! Interface: eno1 
IP address not available for interface.
Warning! Interface: enp4s0f3 
IP address not available for interface.
Warning! Interface: eno4 
IP address not available for interface.
[07:37:54] /opt/dgl/src/rpc/rpc.cc:141: Sender with NetType~socket is created.
[07:37:54] /opt/dgl/src/rpc/rpc.cc:161: Receiver with NetType~socket is created.
Warning! Interface: eno1 
IP address not available for interface.
Warning! Interface: enp4s0f3 
IP address not available for interface.
Warning! Interface: eno4 
IP address not available for interface.
[07:37:54] /opt/dgl/src/rpc/rpc.cc:141: Sender with NetType~socket is created.
[07:37:54] /opt/dgl/src/rpc/rpc.cc:161: Receiver with NetType~socket is created.
Warning! Interface: eno1 
IP address not available for interface.
Warning! Interface: enp4s0f3 
IP address not available for interface.
Warning! Interface: eno4 
IP address not available for interface.
[07:37:54] /opt/dgl/src/rpc/rpc.cc:141: Sender with NetType~socket is created.
[07:37:54] /opt/dgl/src/rpc/rpc.cc:161: Receiver with NetType~socket is created.
Warning! Interface: eno1 
IP address not available for interface.
Warning! Interface: enp4s0f3 
IP address not available for interface.
Warning! Interface: eno4 
IP address not available for interface.
[07:37:54] /opt/dgl/src/rpc/rpc.cc:141: Sender with NetType~socket is created.
[07:37:54] /opt/dgl/src/rpc/rpc.cc:161: Receiver with NetType~socket is created.
Warning! Interface: eno1 
IP address not available for interface.
Warning! Interface: enp4s0f3 
IP address not available for interface.
Warning! Interface: eno4 
IP address not available for interface.
[07:37:54] /opt/dgl/src/rpc/rpc.cc:141: Sender with NetType~socket is created.
[07:37:54] /opt/dgl/src/rpc/rpc.cc:161: Receiver with NetType~socket is created.
Client [35334] waits on 172.16.0.136:60031
Client [35333] waits on 172.16.0.136:44803
Client [46137] waits on 172.16.0.107:47312
Client [46139] waits on 172.16.0.107:36412
Client [46138] waits on 172.16.0.107:58852
Client [46140] waits on 172.16.0.107:59469
Client [35332] waits on 172.16.0.136:34853
Client [35331] waits on 172.16.0.136:39522
Machine (0) group (0) client (0) connect to server successfuly!
Machine (0) group (0) client (3) connect to server successfuly!
Machine (0) group (0) client (6) connect to server successfuly!
Machine (0) group (0) client (7) connect to server successfuly!
Machine (1) group (0) client (10) connect to server successfuly!
Machine (1) group (0) client (8) connect to server successfuly!
Machine (1) group (0) client (9) connect to server successfuly!
Machine (1) group (0) client (15) connect to server successfuly!
Epoch 0: Validation Accuracy 0.44414607948442536
Epoch 0: Validation Accuracy 0.4391946308724832
Epoch 0: Validation Accuracy 0.45610738255033556
Epoch 0: Validation Accuracy 0.16778523489932887
Epoch 0: Validation Accuracy 0.17315436241610738
Epoch 0: Validation Accuracy 0.44483221476510065
Epoch 0: Validation Accuracy 0.1632214765100671
Epoch 0: Validation Accuracy 0.16456375838926174
Client[15] in group[0] is exiting...
Client[3] in group[0] is exiting...
Client[0] in group[0] is exiting...
Client[9] in group[0] is exiting...
Client[6] in group[0] is exiting...
Client[7] in group[0] is exiting...
Client[10] in group[0] is exiting...
Client[8] in group[0] is exiting...
Start to load partition from 4part_data/part0/graph.dgl which is 33557645 bytes. It may take non-trivial time for large partition.
Finished loading partition.
load ogbn-arxiv
Start to create specified graph formats which may take non-trivial time.
Finished creating specified graph formats.
Start to load node data from 4part_data/part0/node_feat.dgl which is 46126812 bytes.
Finished loading node data.
Start to load edge data from 4part_data/part0/edge_feat.dgl which is 24 bytes.
Finished loading edge data.
start graph service on server 0 for part 0
Server is waiting for connections on [172.16.0.107:27000]...
Server (0) shutdown.
Server is exiting...
debug1: client_input_channel_req: channel 0 rtype exit-status reply 0
debug1: client_input_channel_req: channel 0 rtype eow@openssh.com reply 0
debug1: channel 0: free: client-session, nchannels 1
Transferred: sent 3968, received 3836 bytes, in 12.3 seconds
Bytes per second: sent 323.5, received 312.8
debug1: Exit status 0
Start to load partition from 4part_data/part1/graph.dgl which is 35927339 bytes. It may take non-trivial time for large partition.
Finished loading partition.
load ogbn-arxiv
Start to create specified graph formats which may take non-trivial time.
Finished creating specified graph formats.
Start to load node data from 4part_data/part1/node_feat.dgl which is 43795191 bytes.
Finished loading node data.
Start to load edge data from 4part_data/part1/edge_feat.dgl which is 24 bytes.
Finished loading edge data.
start graph service on server 1 for part 1
Server is waiting for connections on [172.16.0.136:28000]...
Server (1) shutdown.
Server is exiting...
Client [46205] waits on 172.16.0.107:50619
Machine (0) group (0) client (4) connect to server successfuly!
Client[4] in group[0] is exiting...
Client [35389] waits on 172.16.0.136:46550
Machine (1) group (0) client (11) connect to server successfuly!
Client[11] in group[0] is exiting...
Client [46191] waits on 172.16.0.107:36444
Machine (0) group (0) client (1) connect to server successfuly!
Client[1] in group[0] is exiting...
Client [46195] waits on 172.16.0.107:54883
Machine (0) group (0) client (5) connect to server successfuly!
Client[5] in group[0] is exiting...
Client [46199] waits on 172.16.0.107:44541
Machine (0) group (0) client (2) connect to server successfuly!
Client[2] in group[0] is exiting...
Client [35399] waits on 172.16.0.136:59361
Machine (1) group (0) client (14) connect to server successfuly!
Client[14] in group[0] is exiting...
Client [35395] waits on 172.16.0.136:49352
Machine (1) group (0) client (12) connect to server successfuly!
Client[12] in group[0] is exiting...
Client [35385] waits on 172.16.0.136:55542
Machine (1) group (0) client (13) connect to server successfuly!
Client[13] in group[0] is exiting...
debug1: client_input_channel_req: channel 0 rtype exit-status reply 0
debug1: client_input_channel_req: channel 0 rtype eow@openssh.com reply 0
debug1: channel 0: free: client-session, nchannels 1
Transferred: sent 3968, received 3836 bytes, in 13.2 seconds
Bytes per second: sent 300.3, received 290.3
debug1: Exit status 0
/home/kai/anaconda3/envs/py376/lib/python3.7/site-packages/torch/distributed/launch.py:188: FutureWarning: The module torch.distributed.launch is deprecated
and will be removed in future. Use torchrun.
Note that --use_env is set by default in torchrun.
If your script expects `--local_rank` argument to be set, please
change it to read from `os.environ['LOCAL_RANK']` instead. See 
https://pytorch.org/docs/stable/distributed.html#launch-utility for 
further instructions

  FutureWarning,
/home/kai/anaconda3/envs/py376/lib/python3.7/site-packages/torch/distributed/launch.py:188: FutureWarning: The module torch.distributed.launch is deprecated
and will be removed in future. Use torchrun.
Note that --use_env is set by default in torchrun.
If your script expects `--local_rank` argument to be set, please
change it to read from `os.environ['LOCAL_RANK']` instead. See 
https://pytorch.org/docs/stable/distributed.html#launch-utility for 
further instructions

  FutureWarning,
debug1: client_input_channel_req: channel 0 rtype exit-status reply 0
debug1: client_input_channel_req: channel 0 rtype eow@openssh.com reply 0
debug1: channel 0: free: client-session, nchannels 1
Transferred: sent 4096, received 9312 bytes, in 17.9 seconds
Bytes per second: sent 229.4, received 521.6
debug1: Exit status 0
debug1: client_input_channel_req: channel 0 rtype exit-status reply 0
debug1: client_input_channel_req: channel 0 rtype eow@openssh.com reply 0
debug1: channel 0: free: client-session, nchannels 1
Transferred: sent 4096, received 9320 bytes, in 17.7 seconds
Bytes per second: sent 232.1, received 528.0
debug1: Exit status 0
```
