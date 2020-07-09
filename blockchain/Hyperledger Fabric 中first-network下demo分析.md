# 一、引子
安装Hyperleder Fabric后都要跑一下demo

在**fabric-sample/first-network**路径下执行了下列命令，来生成、建立网络并执行例子和清除网络：
```
./byfn.sh generate
./byfn.sh up
./byfn.sh down
```

# 二、 byfn.sh脚本分析
打开文件可以看到脚本开头注释：

```
# This script will orchestrate a sample end-to-end execution of the Hyperledger
# Fabric network. 
#         
# The end-to-end verification provisions a sample Fabric network consisting of
# two organizations, each maintaining two peers, and a “solo” ordering service.
#         
# This verification makes use of two fundamental tools, which are necessary to
# create a functioning transactional network with digital signature validation
# and access control:
#            
# * cryptogen - generates the x509 certificates used to identify and
#   authenticate the various components in the network.
# * configtxgen - generates the requisite configuration artifacts for orderer
#   bootstrap and channel creation.
#
# Each tool consumes a configuration yaml file, within which we specify the topology
# of our network (cryptogen) and the location of our certificates for various
# configuration operations (configtxgen).  Once the tools have been successfully run,
# we are able to launch our network.  More detail on the tools and the structure of
# the network will be provided later in this document.  For now, let's get going...
```
这个注释大概意思：
- 这个脚本将完成执行一个Hyperledger Fabric网络中一个端到端的例子。
- 这个例子提供一个样本的Fabric网络，网络中有两个组织，每个组织中有两个节点，使用"solo"排序方式。
- 主要使用了两个必要的工具：
    - cryptogen：用来生成X509证书
    - configtxgen：用来为orderer引导程序和通道创建生成必备的文件
- 每个工具(cryptogen 和 configtxgen)使用一个yaml配置文件，配置文件和操作的路径由我们指定。一旦工具成功运行，我们就可以启动我们的网络。

随后可以看到脚本内容：

```
# prepending $PWD/../bin to PATH to ensure we are picking up the correct binaries
# this may be commented out to resolve installed version of tools if desired
export PATH=${PWD}/../bin:${PWD}:$PATH
export FABRIC_CFG_PATH=${PWD}
export VERBOSE=false
```
- 将 **${PWD}/../bin**添加到PATH确保我们可以找到二进制模块（即fabric中的cryptogen等二进制文件
- 设置了FABRIC_CFG_PATH为当前路径(即$(PWD))

脚本在执行完三行export语句后首先，判断当前操作系统的版本和类型：

```
OS_ARCH=$(echo "$(uname -s | tr '[:upper:]' '[:lower:]' | sed 's/mingw64_nt.*/windows/')-$(uname -m | sed 's/x86_64/amd64/g')" | awk '{print tolower($0)}')
```
之后给超时，延迟，通道名称，各种配置文件名称，链码语言等变量赋默认值：
```
# timeout duration - the duration the CLI should wait for a response from
# another container before giving up
CLI_TIMEOUT=10
# default for delay between commands
CLI_DELAY=3
# channel name defaults to "mychannel"
CHANNEL_NAME="mychannel"
# use this as the default docker-compose yaml definition
COMPOSE_FILE=docker-compose-cli.yaml
#
COMPOSE_FILE_COUCH=docker-compose-couch.yaml
# org3 docker compose file
COMPOSE_FILE_ORG3=docker-compose-org3.yaml
# kafka and zookeeper compose file
COMPOSE_FILE_KAFKA=docker-compose-kafka.yaml
#
# use golang as the default language for chaincode
LANGUAGE=golang
# default image tag
IMAGETAG="latest"
# default consensus type
CONSENSUS_TYPE="solo"
```
结合着跟打印帮助信息的函数定义，可以理解一部分变量的默认值：
```
# Print the usage message
function printHelp() {
  echo "Usage: "
  echo "  byfn.sh <mode> [-c <channel name>] [-t <timeout>] [-d <delay>] [-f <docker-compose-file>] [-s <dbtype>] [-l <language>] [-o <consensus-type>] [-i <imagetag>] [-v]"
  echo "    <mode> - one of 'up', 'down', 'restart', 'generate' or 'upgrade'"
  echo "      - 'up' - bring up the network with docker-compose up"
  echo "      - 'down' - clear the network with docker-compose down"
  echo "      - 'restart' - restart the network"
  echo "      - 'generate' - generate required certificates and genesis block"
  echo "      - 'upgrade'  - upgrade the network from version 1.3.x to 1.4.0"
  echo "    -c <channel name> - channel name to use (defaults to \"mychannel\")"
  echo "    -t <timeout> - CLI timeout duration in seconds (defaults to 10)"
  echo "    -d <delay> - delay duration in seconds (defaults to 3)"
  echo "    -f <docker-compose-file> - specify which docker-compose file use (defaults to docker-compose-cli.yaml)"
  echo "    -s <dbtype> - the database backend to use: goleveldb (default) or couchdb"
  echo "    -l <language> - the chaincode language: golang (default) or node"
  echo "    -o <consensus-type> - the consensus-type of the ordering service: solo (default) or kafka"
  echo "    -i <imagetag> - the tag to be used to launch the network (defaults to \"latest\")"
  echo "    -v - verbose mode"
  echo "  byfn.sh -h (print this message)"
  echo
  echo "Typically, one would first generate the required certificates and "
  echo "genesis block, then bring up the network. e.g.:"
  echo
  echo "        byfn.sh generate -c mychannel"
  echo "        byfn.sh up -c mychannel -s couchdb"
  echo "        byfn.sh up -c mychannel -s couchdb -i 1.4.0"
  echo "        byfn.sh up -l node"
  echo "        byfn.sh down -c mychannel"
  echo "        byfn.sh upgrade -c mychannel"
  echo
  echo "Taking all defaults:"
  echo "        byfn.sh generate"
  echo "        byfn.sh up"
  echo "        byfn.sh down"
}
```
- 可以看到执行脚本的帮助信息，执行脚本命令格式

```
byfn.sh <mode> [-c <channel name>] [-t <timeout>] [-d <delay>] [-f <docker-compose-file>] [-s <dbtype>] [-l <language>] [-i <imagetag>] [-v]
```
- mode:
    - up： 用docker-compose建立网络
    - down：用docker-compose清除网络
    - restart：重启网络
    - generate：生成所需的证书和创世块
    - upgrade：将网络从v1.0升级到v1.1（我用的是1.4版本，这个应该用不到）
- -c 指定通道名称，默认为"mychannel"
- -t 指定CLI超时时间，默认为10s
- -d 指定延迟持续时间，默认为3s
- -s 指定后端数据库，goleveldb(默认) 或者 couchdb
- -l 指定链码语言：golang(默认)，node或者java
- -i 用于启动网络标记，(默认为"lateset")
- -h 打印帮助信息

这里我们省略对参数进行解析，将相关默认值进行改变的代码，直接看解析具体操作的入口：

```
#Create the network using docker compose
if [ "${MODE}" == "up" ]; then
  networkUp
elif [ "${MODE}" == "down" ]; then ## Clear the network
  networkDown
elif [ "${MODE}" == "generate" ]; then ## Generate Artifacts
  generateCerts
  replacePrivateKey
  generateChannelArtifacts
elif [ "${MODE}" == "restart" ]; then ## Restart the network
  networkDown
  networkUp
elif [ "${MODE}" == "upgrade" ]; then ## Upgrade the network from version 1.2.x to 1.3.x
  upgradeNetwork
else
  printHelp
  exit 1
fi
```
- 可以看到up，down，generate,restart和upgrade分别转向具体的方法中去执行

## 2.1 generate
从generate开始入手，可以看到有三个执行方法，并且按照generateCerts ，replacePrivateKey， generateChannelArtifacts的顺序去执行

### 2.1.1 generateCerts
 
首先找到generateCerts方法：

```
function generateCerts() {
  which cryptogen
  if [ "$?" -ne 0 ]; then
    echo "cryptogen tool not found. exiting"
    exit 1
  fi
  echo
  echo "##########################################################"
  echo "##### Generate certificates using cryptogen tool #########"
  echo "##########################################################"

  if [ -d "crypto-config" ]; then
    rm -Rf crypto-config
  fi
  set -x
  cryptogen generate --config=./crypto-config.yaml
  res=$?
  set +x
  if [ $res -ne 0 ]; then
    echo "Failed to generate certificates..."
    exit 1
  fi
  echo
}
```
- which cryptogen：先去找cryptogen，如果返回不是0(即没有找到)，则打印提示并退出
- if [-d "crypto-config"]：如果有这个目录，则删除这个目录
- cryptogen generate --config=./crypto-config.yaml：使用指定的yaml配置文件来生成相关的证书，生成到当前目录下的crypto-config文件夹下

#### crypto-config.yaml配置文件
这个crypto-config.yaml配置文件用来帮助cryptogen生成证书，可以打开看到具体信息：
- OrdererOrgs：对排序节点组织的相关信息定义，有着节点名称，节点域名，主机名的三个具体信息：
    ```
    OrdererOrgs:
      - Name: Orderer
        Domain: example.com
        Specs:
          - Hostname: orderer
    ```
- PeerOrgs：对peer节点组织的相关信息定义，节点组织名称，域名和模板信息、模板信息中指定允许通过该模板Template中的Count定义创建0到Count-1个所属组织节点或主机；在Users中的Count定义除Admin之外创建用户的数量
```
PeerOrgs:
  - Name: Org1
    Domain: org1.example.com
    EnableNodeOUs: true
    Template:
        Count: 2
    Users:
        Count: 1
  - Name: Org2
    Domain: org2.example.com
    EnableNodeOUs: true
    Template:
      Count: 2
    Users:
      Count: 1
```

### 2.1.2 replacePrivateKey
接着generate的三个方法中第一个执行完后，执行replacePrivateKey这一方法：

```
function replacePrivateKey() {
  # sed on MacOSX does not support -i flag with a null extension. We will use
  # 't' for our back-up's extension and delete it at the end of the function
  ARCH=$(uname -s | grep Darwin)
  if [ "$ARCH" == "Darwin" ]; then
    OPTS="-it"
  else
    OPTS="-i"
  fi

  # Copy the template to the file that will be modified to add the private key
  cp docker-compose-e2e-template.yaml docker-compose-e2e.yaml

  # The next steps will replace the template's contents with the
  # actual values of the private key file names for the two CAs.
  CURRENT_DIR=$PWD
  cd crypto-config/peerOrganizations/org1.example.com/ca/
  PRIV_KEY=$(ls *_sk)
  cd "$CURRENT_DIR"
  sed $OPTS "s/CA1_PRIVATE_KEY/${PRIV_KEY}/g" docker-compose-e2e.yaml
  cd crypto-config/peerOrganizations/org2.example.com/ca/
  PRIV_KEY=$(ls *_sk)
  cd "$CURRENT_DIR"
  sed $OPTS "s/CA2_PRIVATE_KEY/${PRIV_KEY}/g" docker-compose-e2e.yaml
  # If MacOSX, remove the temporary backup of the docker-compose file
  if [ "$ARCH" == "Darwin" ]; then
    rm docker-compose-e2e.yamlt
  fi
}
```
- if["$ARCH" == "Darwin"]：如果是在MACOS系统下，sed命令不支持 -i这一参数，所以变成 -it，我执行所在操作系统是Linux所以不牵扯
- cp docker-compose-e2e-template.yaml docker-compose-e2e.yaml：把这个模板配置文件拷贝出来，目的是给配置文件中加上私钥(也就是刚才cryptogen生成文件中的相关部分)
- sed $OPTS "s/CA1_PRIVATE_KEY/${PRIV_KEY/G" docker-compose-e2e.yaml：这里可以看到在yaml配置文件中，把CA1_PRIVATE_KEY替换成crypto-config/peerOrganizations/org1.example.com/ca/目录下以"_sk"结尾的文件名
- 同样的把CA2_PRIVATE_KEY替换成组织2中CA的私钥
- 最终生成的文件在当前目录下命名为docker-compose-e2e.yaml，这一配置文件中定义了CA的证书文件和秘钥文件，并且通过commond启动了Fabric-CA服务器：
    ```
    command: sh -c 'fabric-ca-server start --ca.certfile /etc/hyperledger/fabric-ca-server-config/ca.org1.example.com-cert.pem --ca.keyfile /etc/hyperledger/fabric-ca-server-config/25b3b06bba684dd461670f2888a2e71f93e5094e03a486e638c1ebe4db1bb4fa_sk -b admin:adminpw -d'
    ```
    
 
### 2.1.3 generateChannelArtifacts
在generate中第二个方法执行过后，开始执行generateChannelArtifacts方法，由于这个方法比较长，就不全部复制过来：
- 首先通过which configtxgen来查询必要二进制文件是否存在,接下来的操作都要围绕这一二进制文件操作，并且需要使用configtx.yaml这一配置文件。
- 随后通过configtxgen，根据我们选定的不同排序方式(默认kafka)，来生成创世块：
```
 if [ "$CONSENSUS_TYPE" == "solo" ]; then
    configtxgen -profile TwoOrgsOrdererGenesis -channelID byfn-sys-channel -outputBlock ./channel-artifacts/genesis.block
 elif [ "$CONSENSUS_TYPE" == "kafka" ]; then
    configtxgen -profile SampleDevModeKafka -channelID byfn-sys-channel -outputBlock ./channel-artifacts/genesis.block
```
- 之后还通过configtxgen创建Channel.tx文件(通道配置文件），channel可以叫做账本也可以叫做通道
```
 configtxgen -profile TwoOrgsChannel -outputCreateChannelTx ./channel-artifacts/channel.tx -channelID $CHANNEL_NAME
```
- 之后生成组织1和组织2的锚点文件：
```
 configtxgen -profile TwoOrgsChannel -outputAnchorPeersUpdate ./channel-artifacts/Org1MSPanchors.tx -channelID $CHANNEL_NAME -asOrg Org1MSP
 configtxgen -profile TwoOrgsChannel -outputAnchorPeersUpdate ./channel-artifacts/Org2MSPanchors.tx -channelID $CHANNEL_NAME -asOrg Org2MSP
```
- 总结一下：generateChannelArtifacts方法使用configtxgen根据configtx.yaml配置文件，创建了一个创始块，一个Channel和两个锚点文件

#### configtx.yaml配置文件
这里放一个configtx.yaml配置文件并配上相应注释：
```
Profiles:
    # 以下部分定义了整个系统的配置信息
    # 组织定义标识符，可自定义，命令中的-profile参数对应该标识符
    TestTwoOrgsOrdererGenesis:
        # orderer 配置属性，系统关键字不得更改
        Orderer:
            #引用下面名为OrdererDefaults的属性
            <<: *OrdererDefaults
            Organizations:
                #引用下面的名为OrdererOrg的属性
                - *OrdererOrg
        # 定义了系统中包含的组织
        Consortiums:
            SampleConsortium:
                #系统中包含的组织
                Organizations:
                    - *Org1   #引用了下文定义配置
                    - *Org2

    # 以下内容为channel的配置信息
    # 通道定义标识符，可自定义
    TestTwoOrgsChannel:
        Consortium: SampleConsortium
        Application:
            <<: *ApplicationDefaults
            Organizations:
                - *Org1
                - *Org2

# orderer节点相关信息
Organizations:
    # orderer节点配置信息
    - &OrdererOrg
        # orderer节点名称
        Name: OrdererOrg
        # orderer节点编号
        ID: OrdererMSP
        # mps文件夹路径
        MSPDir: msp
    
    # Orderer节点中包含的组织，如果有多个组织可以配置多个
    - &Org1
        Name: Org1MSP     # 组织名称
        ID: Org1MSP       # 组织编号
        MSPDir: msp       # 组织msp文件名
        AnchorPeers:      # 组织的访问域名和端口
            - Host: peer0.org1.testcryptogen.com
              Port: 7051

    - &Org2
        Name: Org2MSP
        ID: Org2MSP
        MSPDir: msp
        AnchorPeers:
            - Host: peer0.org2.testcryptogen.com
              Port: 7051

# orderer节点的配置信息
Orderer: &OrdererDefaults
    OrdererType: solo           # orderer节点共识办法
    Addresses:
        - orderer.testcryptogen.com:7050  # orderer监听的地址
    BatchTimeout: 2s

    BatchSize:
        MaxMessageCount: 10
        AbsoluteMaxBytes: 98 MB
        PreferredMaxBytes: 512KB

    Kafka:
        Brokers:
            - 127.0.0.1:9092
    Organizations:

Application: &ApplicationDefaults

    Organizations:
```
- Profiles：定义整个系统的配置信息
- Organizations：定义组织节点信息
- Orderer：定义Orderer节点信息
- Application：定义了将编码转换为配置的值或者应用程序相关信息，不需要配置即可

### 2.1.4 generate小结
- 根据顺序执行generateCerts ，replacePrivateKey， generateChannelArtifacts三个方法
- generateCerts中使用cryptogen根据crypto-config.yaml配置文件生成证书和秘钥文件，放在crypto-config文件夹下
- replacePrivateKey通过将生成的证书秘钥文件，替换docker-compose-e2e.yaml文件中相关的参数
- generateChannelArtifacts方法使用configtxgen根据configtx.yaml配置文件，创建了一个创始块，一个Channel.tx文件和两个锚点文件

## 2.2 up
可以从脚本中看到如果执行up，转到执行networkUp这一方法
```
if [ "${MODE}" == "up" ]; then
  networkUp
```
在networkUp中首先执行checkPrereqs方法

### 2.2.1 checkPrereqs
在这个方法中进行一些基本的完整性检查，确认可以使用符合要求版本的二进制文件和docker镜像。 

以后还可以在这个方法中添加额外检查，比如是否存在合适版本的GO或者一些其他项目。


- checkPrereqs方法，首先确定configtxlator这一二进制文件的版本，再通过docker镜像，执行peer version得到peer的版本，如果两个版本号不同则给出警告：
```
LOCAL_VERSION=$(configtxlator version | sed -ne 's/ Version: //p')
DOCKER_IMAGE_VERSION=$(docker run --rm hyperledger/fabric-tools:$IMAGETAG peer version | sed -ne 's/ Version: //p' | head -1)

echo "LOCAL_VERSION=$LOCAL_VERSION"
echo "DOCKER_IMAGE_VERSION=$DOCKER_IMAGE_VERSION"

if [ "$LOCAL_VERSION" != "$DOCKER_IMAGE_VERSION" ]; then
    echo "=================== WARNING ==================="
    echo "  Local fabric binaries and docker images are  "
    echo "  out of  sync. This may cause problems.       "
    echo "==============================================="
fi
```
- 随后遍历了BALCKLISTED_VERSIONS，这里包括着不支持这个demo的fabric版本，并且跟我们的configtxlator和peer版本进行比较，如果我们的configtxlator和peer版本是不支持这个domo的，此时输出错误并退出：
```
BLACKLISTED_VERSIONS="^1\.0\. ^1\.1\.0-preview ^1\.1\.0-alpha"
...
for UNSUPPORTED_VERSION in $BLACKLISTED_VERSIONS; do
echo "$LOCAL_VERSION" | grep -q $UNSUPPORTED_VERSION
if [ $? -eq 0 ]; then
  echo "ERROR! Local Fabric binary version of $LOCAL_VERSION does not match this newer version of BYFN and is unsupported. Either move to a later version of Fabric or checkout an earlier version of fabric-samples."
  exit 1
fi

echo "$DOCKER_IMAGE_VERSION" | grep -q $UNSUPPORTED_VERSION
if [ $? -eq 0 ]; then
  echo "ERROR! Fabric Docker image version of $DOCKER_IMAGE_VERSION does not match this newer version of BYFN and is unsupported. Either move to a later version of Fabric or checkout an earlier version of fabric-samples."
  exit 1
fi
done
```

### 2.2.2 start network
执行完checkPrereqs后，判断如果还没有crypto-config文件夹，那么证明我们在执行./byfn.sh up之前没有执行./byfn.sh generate
，所以需要执行generate下的三个方法：generateCerts ，replacePrivateKey，generateChannelArtifacts：

```
if [ ! -d "crypto-config" ]; then
    generateCerts
    replacePrivateKey
    generateChannelArtifacts
fi
```

随后根据是否指定了后端数据库是couchdb(默认goleveldb)，是否指定了排序方式是kafka(默认solo)，使用docker-compose 命令中传入对应的yaml配置文件
```
if [ "${IF_COUCHDB}" == "couchdb" ]; then
    if [ "$CONSENSUS_TYPE" == "kafka" ]; then
        IMAGE_TAG=$IMAGETAG docker-compose -f $COMPOSE_FILE -f $COMPOSE_FILE_KAFKA -f $COMPOSE_FILE_COUCH up -d 2>&1
    else
        IMAGE_TAG=$IMAGETAG docker-compose -f $COMPOSE_FILE -f $COMPOSE_FILE_COUCH up -d 2>&1
    fi
else
    if [ "$CONSENSUS_TYPE" == "kafka" ]; then
        IMAGE_TAG=$IMAGETAG docker-compose -f $COMPOSE_FILE -f $COMPOSE_FILE_KAFKA up -d 2>&1
    else
        IMAGE_TAG=$IMAGETAG docker-compose -f $COMPOSE_FILE up -d 2>&1
    fi
fi
if [ $? -ne 0 ]; then
    echo "ERROR !!!! Unable to start network"
    exit 1
fi
```
这里COMPOSE_FILE=docker-compose-cli.yaml, 下面看一看这个容器启动配置文件：

#### docker-compose-cli.yaml

```
version: '2'

volumes:
  orderer.example.com:
  peer0.org1.example.com:
  peer1.org1.example.com:
  peer0.org2.example.com:
  peer1.org2.example.com:

networks:
  byfn:

services:

  orderer.example.com:
    extends:
      file:   base/docker-compose-base.yaml
      service: orderer.example.com
    container_name: orderer.example.com
    networks:
      - byfn

  peer0.org1.example.com:
    container_name: peer0.org1.example.com
    extends:
      file:  base/docker-compose-base.yaml
      service: peer0.org1.example.com
    networks:
      - byfn

  peer1.org1.example.com:
    container_name: peer1.org1.example.com
    extends:
      file:  base/docker-compose-base.yaml
      service: peer1.org1.example.com
    networks:
      - byfn

  peer0.org2.example.com:
    container_name: peer1.org2.example.com
    extends:
      file:  base/docker-compose-base.yaml
      service: peer1.org2.example.com
    networks:
      - byfn

  cli:
    container_name: cli
    image: hyperledger/fabric-tools:$IMAGE_TAG
    tty: true
    stdin_open: true
    environment:
      - GOPATH=/opt/gopath
      - CORE_VM_ENDPOINT=unix:///host/var/run/docker.sock
      #- FABRIC_LOGGING_SPEC=DEBUG
      - FABRIC_LOGGING_SPEC=INFO
      - CORE_PEER_ID=cli
      - CORE_PEER_ADDRESS=peer0.org1.example.com:7051
      - CORE_PEER_LOCALMSPID=Org1MSP
      - CORE_PEER_TLS_ENABLED=true
      - CORE_PEER_TLS_CERT_FILE=/opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/server.crt
      - CORE_PEER_TLS_KEY_FILE=/opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/server.key
      - CORE_PEER_TLS_ROOTCERT_FILE=/opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt
      - CORE_PEER_MSPCONFIGPATH=/opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp
    working_dir: /opt/gopath/src/github.com/hyperledger/fabric/peer
    command: /bin/bash
    volumes:
        - /var/run/:/host/var/run/
        - ./../chaincode/:/opt/gopath/src/github.com/chaincode
        - ./crypto-config:/opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/
        - ./scripts:/opt/gopath/src/github.com/hyperledger/fabric/peer/scripts/
        - ./channel-artifacts:/opt/gopath/src/github.com/hyperledger/fabric/peer/channel-artifacts
    depends_on:
      - orderer.example.com
      - peer0.org1.example.com
      - peer1.org1.example.com
      - peer0.org2.example.com
      - peer1.org2.example.com
    networks:
      - byfn
```
- 可以看到排序服务节点继承了base/docker-compose-base.yaml中的orderer.example.com属性；四个peer节点继承了base/docker-compose-base.yaml中名称对应的属性
- docker-compose-base.yaml中orderer.example.com的相关配置信息：
    ```
      orderer.example.com:
        container_name: orderer.example.com
        image: hyperledger/fabric-orderer:$IMAGE_TAG
        environment:
          - FABRIC_LOGGING_SPEC=INFO
          - ORDERER_GENERAL_LISTENADDRESS=0.0.0.0
          - ORDERER_GENERAL_GENESISMETHOD=file
          - ORDERER_GENERAL_GENESISFILE=/var/hyperledger/orderer/orderer.genesis.block
          - ORDERER_GENERAL_LOCALMSPID=OrdererMSP
          - ORDERER_GENERAL_LOCALMSPDIR=/var/hyperledger/orderer/msp
          # enabled TLS
          - ORDERER_GENERAL_TLS_ENABLED=true
          - ORDERER_GENERAL_TLS_PRIVATEKEY=/var/hyperledger/orderer/tls/server.key
          - ORDERER_GENERAL_TLS_CERTIFICATE=/var/hyperledger/orderer/tls/server.crt
          - ORDERER_GENERAL_TLS_ROOTCAS=[/var/hyperledger/orderer/tls/ca.crt]
          - ORDERER_KAFKA_TOPIC_REPLICATIONFACTOR=1
          - ORDERER_KAFKA_VERBOSE=true
        working_dir: /opt/gopath/src/github.com/hyperledger/fabric
        command: orderer
        volumes:
        - ../channel-artifacts/genesis.block:/var/hyperledger/orderer/orderer.genesis.block
        - ../crypto-config/ordererOrganizations/example.com/orderers/orderer.example.com/msp:/var/hyperledger/orderer/msp
        - ../crypto-config/ordererOrganizations/example.com/orderers/orderer.example.com/tls/:/var/hyperledger/orderer/tls
        - orderer.example.com:/var/hyperledger/production/orderer
        ports:
          - 7050:7050
    ```
    - environment：表示当前所配置的外界的环境变量
    - working_dir：表示当前容器启动后的工作路径
    - volumes：外界物理机路径映射到容器内路径
    - ports：当前容器启动后映射到物理机上的端口号
- 在docker-compose-cli.yaml配置文件中还有一个cli服务，这个服务是对peer和orderer这写服务进行数据维护和管理的客户端或SDK的相关服务，这个服务中除上面几个属性外，还有一个depends_on，指当前容器启动依赖的容器启动项
- 经过docker-compose的操作后，网络就已经建立起来了

### 2.2.3 script.sh
建立起来网络后，执行script.sh脚本，这一脚本中真正地对Peer节点、Channel和智能合约集合操作演示：
```
docker exec cli scripts/script.sh $CHANNEL_NAME $CLI_DELAY $LANGUAGE $CLI_TIMEOUT $VERBOSE
if [ $? -ne 0 ]; then
    echo "ERROR !!!! Test failed"
    exit 1
fi
```
#### script.sh
打开script.sh脚本，可以看到根据命令传入参数，对通道名、延迟时间等进行赋值，和这些的默认值：
```
CHANNEL_NAME="$1"
DELAY="$2"
LANGUAGE="$3"
TIMEOUT="$4"
VERBOSE="$5"
: ${CHANNEL_NAME:="mychannel"}
: ${DELAY:="3"}
: ${LANGUAGE:="golang"}
: ${TIMEOUT:="10"}
```
首先看一下script.sh脚本中方法执行顺序：

```
## Create channel
echo "Creating channel..."
createChannel

## Join all the peers to the channel
echo "Having all peers join the channel..."
joinChannel

## Set the anchor peers for each org in the channel
echo "Updating anchor peers for org1..."
updateAnchorPeers 0 1
echo "Updating anchor peers for org2..."
updateAnchorPeers 0 2

## Install chaincode on peer0.org1 and peer0.org2
echo "Installing chaincode on peer0.org1..."
installChaincode 0 1
echo "Install chaincode on peer0.org2..."
installChaincode 0 2

# Instantiate chaincode on peer0.org2
echo "Instantiating chaincode on peer0.org2..."
instantiateChaincode 0 2

# Query chaincode on peer0.org1
echo "Querying chaincode on peer0.org1..."
chaincodeQuery 0 1 100

# Invoke chaincode on peer0.org1 and peer0.org2
echo "Sending invoke transaction on peer0.org1 peer0.org2..."
chaincodeInvoke 0 1 0 2

## Install chaincode on peer1.org2
echo "Installing chaincode on peer1.org2..."
installChaincode 1 2

# Query on chaincode on peer1.org2, check if the result is 90
echo "Querying chaincode on peer1.org2..."
chaincodeQuery 1 2 90
```
可以看到脚本执行步骤为：
1. createChannel：根据之前在byfn.sh脚本中的generateChannelArtifacts方法生成的channel.tx文件创建channel：
    ```
    peer channel create -o orderer.example.com:7050 -c $CHANNEL_NAME -f ./channel-artifacts/channel.tx >&log.txt
    ```
2. joinChannel：将Peer节点加入指定的Channel
    ```
    joinChannel () {
            for org in 1 2; do
                for peer in 0 1; do
                    joinChannelWithRetry $peer $org
                    ...
    joinChannelWithRetry() {
        ...
        peer channel join -b $CHANNEL_NAME.block >&log.txt
        ...
    ```
3. updateAnchorPeers：为channel中的每个组织设置Anchor节点(锚节点)。(updateAnchorPeers在scripts/utils.sh中定义)
    ```
    upgradeChaincode() {
        PEER=$1
        ORG=$2
        ...
        peer channel update -o orderer.example.com:7050 -c $CHANNEL_NAME -f ./channel-artifacts/${CORE_PEER_LOCALMSPID}anchors.tx >&log.txt
        ...
    ```

4. installChaincode：在peer0.org1和peer0.org2上安装智能合约
    ```
    installChaincode() {
      PEER=$1
      ORG=$2
      setGlobals $PEER $ORG
      ...
      peer chaincode install -n mycc -v ${VERSION} -l ${LANGUAGE} -p ${CC_SRC_PATH} >&log.txt
      ...
    ```
5. instantiateChaincode：在peer0.org2上对智能合约进行实例化操作
    ```
    instantiateChaincode() {
      PEER=$1
      ORG=$2
      setGlobals $PEER $ORG
      ...
      peer chaincode instantiate -o orderer.example.com:7050 -C $CHANNEL_NAME -n mycc -l ${LANGUAGE} -v ${VERSION} -c '{"Args":["init","a","100","b","200"]}' -P "AND ('Org1MSP.peer','Org2MSP.peer')" >&log.txt
      ...
    ```

6. chaincodeQuery：在peer0.org1上执行智能合约中查询方法
    ```
    chaincodeQuery() {
      PEER=$1
      ORG=$2
      setGlobals $PEER $ORG
      ...
      peer chaincode query -C $CHANNEL_NAME -n mycc -c '{"Args":["query","a"]}' >&log.txt
      ...
    ```

7. chaincodeInvoke：在peer0.org1和peer0.org2上进行交易
    ```
    chaincodeInvoke() {
      ...
      peer chaincode invoke -o orderer.example.com:7050 -C $CHANNEL_NAME -n mycc $PEER_CONN_PARMS -c '{"Args":["invoke","a","b","10"]}' >&log.txt
      ...
    ```

8. installChaincode：在peer1.org2上安装智能合约
9. chaincodeQuery：在peer1.org2上执行智能合约中查询方法

至此./byfn.sh up就执行完毕了