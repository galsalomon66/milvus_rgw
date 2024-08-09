#!/bin/bash

cd "$(git rev-parse --show-toplevel)/config"

[ -z ${S3_ENDPOINT_IP} ] && echo "missing end-variable S3_ENDPOINT_IP" && exit
[ -z ${S3_ENDPOINT_PORT} ] && echo "missing end-variable S3_ENDPOINT_PORT" && exit
[ -z ${S3_ACCESS_KEY} ] && echo missing end-variable S3_ACCESS_KEY && exit
[ -z ${S3_SECRET_KEY} ] && echo missing end-variable S3_SECRET_KEY && exit

cat << EOF > milvus.yaml
# Licensed to the LF AI & Data foundation under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Related configuration of etcd, used to store Milvus metadata & service discovery.
etcd:
  endpoints: localhost:2379
  rootPath: by-dev # The root path where data is stored in etcd
  metaSubPath: meta # metaRootPath = rootPath + '/' + metaSubPath
  kvSubPath: kv # kvRootPath = rootPath + '/' + kvSubPath
  log:
    level: info # Only supports debug, info, warn, error, panic, or fatal. Default 'info'.
    # path is one of:
    #  - "default" as os.Stderr,
    #  - "stderr" as os.Stderr,
    #  - "stdout" as os.Stdout,
    #  - file path to append server logs to.
    # please adjust in embedded Milvus: /tmp/milvus/logs/etcd.log
    path: stdout
  ssl:
    enabled: false # Whether to support ETCD secure connection mode
    tlsCert: /path/to/etcd-client.pem # path to your cert file
    tlsKey: /path/to/etcd-client-key.pem # path to your key file
    tlsCACert: /path/to/ca.pem # path to your CACert file
    # TLS min version
    # Optional values: 1.0, 1.1, 1.2, 1.3。
    # We recommend using version 1.2 and above.
    tlsMinVersion: 1.3
  requestTimeout: 10000 # Etcd operation timeout in milliseconds
  use:
    embed: false # Whether to enable embedded Etcd (an in-process EtcdServer).
  data:
    dir: default.etcd # Embedded Etcd only. please adjust in embedded Milvus: /tmp/milvus/etcdData/
  auth:
    enabled: false # Whether to enable authentication
    userName:  # username for etcd authentication
    password:  # password for etcd authentication

metastore:
  type: etcd # Default value: etcd, Valid values: [etcd, tikv] 

# Related configuration of tikv, used to store Milvus metadata.
# Notice that when TiKV is enabled for metastore, you still need to have etcd for service discovery.
# TiKV is a good option when the metadata size requires better horizontal scalability.
tikv:
  endpoints: 127.0.0.1:2389 # Note that the default pd port of tikv is 2379, which conflicts with etcd.
  rootPath: by-dev # The root path where data is stored in tikv
  metaSubPath: meta # metaRootPath = rootPath + '/' + metaSubPath
  kvSubPath: kv # kvRootPath = rootPath + '/' + kvSubPath
  requestTimeout: 10000 # ms, tikv request timeout
  snapshotScanSize: 256 # batch size of tikv snapshot scan
  ssl:
    enabled: false # Whether to support TiKV secure connection mode
    tlsCert:  # path to your cert file
    tlsKey:  # path to your key file
    tlsCACert:  # path to your CACert file

localStorage:
  path: /var/lib/milvus/data/ # please adjust in embedded Milvus: /tmp/milvus/data/

# Related configuration of MinIO/S3/GCS or any other service supports S3 API, which is responsible for data persistence for Milvus.
# We refer to the storage service as MinIO/S3 in the following description for simplicity.
minio:
  #address: 192.168.208.1 # Address of MinIO/S3
  address: ${S3_ENDPOINT_IP} # Address of CEPH/S3 localhost (NOTE: the milvus runs in a container it needs to access the host)
  #port: 9000 # Port of MinIO/S3
  port: ${S3_ENDPOINT_PORT} # Port of CEPH/S3
  #accessKeyID: 1p9PF799pr6YZI1E04nx # accessKeyID of MinIO/S3
  #secretAccessKey: NRwWgG24xx4ZHfS0eDPmr3R1ARdfnryJ05wXYGr9 # MinIO/S3 encryption string
  accessKeyID: ${S3_ACCESS_KEY:-b2345678901234567890} # accessKeyID of CEPH/S3
  secretAccessKey: ${S3_SECRET_KEY:-b234567890123456789012345678901234567890} # CEPH/S3 encryption string
  useSSL: false # Access to MinIO/S3 with SSL
  ssl:
    tlsCACert: /path/to/public.crt # path to your CACert file
  bucketName: hive # Bucket name in CEPH/S3, Milvus writes its files in that bucket
  rootPath: files # The root path(under bucket-name) where the message is stored in CEPH/S3
  # Whether to useIAM role to access S3/GCS instead of access/secret keys
  # For more information, refer to
  # aws: https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_use.html
  # gcp: https://cloud.google.com/storage/docs/access-control/iam
  # aliyun (ack): https://www.alibabacloud.com/help/en/container-service-for-kubernetes/latest/use-rrsa-to-enforce-access-control
  # aliyun (ecs): https://www.alibabacloud.com/help/en/elastic-compute-service/latest/attach-an-instance-ram-role
  useIAM: false
  # Cloud Provider of S3. Supports: "aws", "gcp", "aliyun".
  # You can use "aws" for other cloud provider supports S3 API with signature v4, e.g.: minio
  # You can use "gcp" for other cloud provider supports S3 API with signature v2
  # You can use "aliyun" for other cloud provider uses virtual host style bucket
  # When useIAM enabled, only "aws", "gcp", "aliyun" is supported for now
  cloudProvider: aws
  # Custom endpoint for fetch IAM role credentials. when useIAM is true & cloudProvider is "aws".
  # Leave it empty if you want to use AWS default endpoint
  iamEndpoint: 
  logLevel: debug # Log level for aws sdk log. Supported level:  off, fatal, error, warn, info, debug, trace
  region:  # Specify minio storage system location region
  useVirtualHost: false # Whether use virtual host mode for bucket
  requestTimeoutMs: 10000 # minio timeout for request time in milliseconds
  listObjectsMaxKeys: 0 # The maximum number of objects requested per batch in minio ListObjects rpc, 0 means using oss client by default, decrease these configration if ListObjects timeout

# Milvus supports four MQ: rocksmq(based on RockDB), natsmq(embedded nats-server), Pulsar and Kafka.
# You can change your mq by setting mq.type field.
# If you don't set mq.type field as default, there is a note about enabling priority if we config multiple mq in this file.
# 1. standalone(local) mode: rocksmq(default) > natsmq > Pulsar > Kafka
# 2. cluster mode:  Pulsar(default) > Kafka (rocksmq and natsmq is unsupported in cluster mode)
mq:
  # Default value: "default"
  # Valid values: [default, pulsar, kafka, rocksmq, natsmq]
  type: default
  enablePursuitMode: true # Default value: "true"
  pursuitLag: 10 # time tick lag threshold to enter pursuit mode, in seconds
  pursuitBufferSize: 8388608 # pursuit mode buffer size in bytes
  mqBufSize: 16 # MQ client consumer buffer length

# Related configuration of pulsar, used to manage Milvus logs of recent mutation operations, output streaming log, and provide log publish-subscribe services.
pulsar:
  address: localhost # Address of pulsar
  port: 6650 # Port of Pulsar
  webport: 80 # Web port of pulsar, if you connect directly without proxy, should use 8080
  maxMessageSize: 5242880 # 5 * 1024 * 1024 Bytes, Maximum size of each message in pulsar.
  tenant: public
  namespace: default
  requestTimeout: 60 # pulsar client global request timeout in seconds
  enableClientMetrics: false # Whether to register pulsar client metrics into milvus metrics path.

# If you want to enable kafka, needs to comment the pulsar configs
# kafka:
#   brokerList: 
#   saslUsername: 
#   saslPassword: 
#   saslMechanisms: 
#   securityProtocol: 
#   ssl:
#     enabled: false # whether to enable ssl mode
#     tlsCert:  # path to client's public key (PEM) used for authentication
#     tlsKey:  # path to client's private key (PEM) used for authentication
#     tlsCaCert:  # file or directory path to CA certificate(s) for verifying the broker's key
#     tlsKeyPassword:  # private key passphrase for use with ssl.key.location and set_ssl_cert(), if any
#   readTimeout: 10

rocksmq:
  # The path where the message is stored in rocksmq
  # please adjust in embedded Milvus: /tmp/milvus/rdb_data
  path: /var/lib/milvus/rdb_data
  lrucacheratio: 0.06 # rocksdb cache memory ratio
  rocksmqPageSize: 67108864 # 64 MB, 64 * 1024 * 1024 bytes, The size of each page of messages in rocksmq
  retentionTimeInMinutes: 4320 # 3 days, 3 * 24 * 60 minutes, The retention time of the message in rocksmq.
  retentionSizeInMB: 8192 # 8 GB, 8 * 1024 MB, The retention size of the message in rocksmq.
  compactionInterval: 86400 # 1 day, trigger rocksdb compaction every day to remove deleted data
  compressionTypes: 0,0,7,7,7 # compaction compression type, only support use 0,7. 0 means not compress, 7 will use zstd. Length of types means num of rocksdb level.

# natsmq configuration.
# more detail: https://docs.nats.io/running-a-nats-service/configuration
natsmq:
  server:
    port: 4222 # Port for nats server listening
    storeDir: /var/lib/milvus/nats # Directory to use for JetStream storage of nats
    maxFileStore: 17179869184 # Maximum size of the 'file' storage
    maxPayload: 8388608 # Maximum number of bytes in a message payload
    maxPending: 67108864 # Maximum number of bytes buffered for a connection Applies to client connections
    initializeTimeout: 4000 # waiting for initialization of natsmq finished
    monitor:
      trace: false # If true enable protocol trace log messages
      debug: false # If true enable debug log messages
      logTime: true # If set to false, log without timestamps.
      logFile: /tmp/milvus/logs/nats.log # Log file path relative to .. of milvus binary if use relative path
      logSizeLimit: 536870912 # Size in bytes after the log file rolls over to a new one
    retention:
      maxAge: 4320 # Maximum age of any message in the P-channel
      maxBytes:  # How many bytes the single P-channel may contain. Removing oldest messages if the P-channel exceeds this size
      maxMsgs:  # How many message the single P-channel may contain. Removing oldest messages if the P-channel exceeds this limit

# Related configuration of rootCoord, used to handle data definition language (DDL) and data control language (DCL) requests
rootCoord:
  dmlChannelNum: 16 # The number of dml channels created at system startup
  maxPartitionNum: 4096 # Maximum number of partitions in a collection
  minSegmentSizeToEnableIndex: 1024 # It's a threshold. When the segment size is less than this value, the segment will not be indexed
  enableActiveStandby: false
  maxDatabaseNum: 64 # Maximum number of database
  maxGeneralCapacity: 65536 # upper limit for the sum of of product of partitionNumber and shardNumber
  gracefulStopTimeout: 5 # seconds. force stop node without graceful stop
  ip:  # if not specified, use the first unicastable address
  port: 53100
  grpc:
    serverMaxSendSize: 536870912
    serverMaxRecvSize: 268435456
    clientMaxSendSize: 268435456
    clientMaxRecvSize: 536870912

# Related configuration of proxy, used to validate client requests and reduce the returned results.
proxy:
  timeTickInterval: 200 # ms, the interval that proxy synchronize the time tick
  healthCheckTimeout: 3000 # ms, the interval that to do component healthy check
  healthCheckTimetout: 3000 # ms, the interval that to do component healthy check
  msgStream:
    timeTick:
      bufSize: 512
  maxNameLength: 255 # Maximum length of name for a collection or alias
  # Maximum number of fields in a collection.
  # As of today (2.2.0 and after) it is strongly DISCOURAGED to set maxFieldNum >= 64.
  # So adjust at your risk!
  maxFieldNum: 64
  maxVectorFieldNum: 4 # Maximum number of vector fields in a collection.
  maxShardNum: 16 # Maximum number of shards in a collection
  maxDimension: 32768 # Maximum dimension of a vector
  # Whether to produce gin logs.\n
  # please adjust in embedded Milvus: false
  ginLogging: true
  ginLogSkipPaths: / # skip url path for gin log
  maxTaskNum: 1024 # max task number of proxy task queue
  accessLog:
    enable: true  # if use access log
    minioEnable: true # if upload sealed access log file to minio
    localPath: /tmp/milvus_access
    filename:  # Log filename, leave empty to use stdout.
    maxSize: 64 # Max size for a single file, in MB.
    cacheSize: 10240 # Size of log of memory cache, in B
    rotatedTime: 0 # Max time for single access log file in seconds
    remotePath: access_log/ # File path in minIO
    remoteMaxTime: 0 # Max time for log file in minIO, in hours
    formatters:
      base:
        format: "[\$time_now] [ACCESS] <\$user_name: \$user_addr> \$method_name [status: \$method_status] [code: \$error_code] [sdk: \$sdk_version] [msg: \$error_msg] [traceID: \$trace_id] [timeCost: \$time_cost]"
      query:
        format: "[\$time_now] [ACCESS] <\$user_name: \$user_addr> \$method_name [status: \$method_status] [code: \$error_code] [sdk: \$sdk_version] [msg: \$error_msg] [traceID: \$trace_id] [timeCost: \$time_cost] [database: \$database_name] [collection: \$collection_name] [partitions: \$partition_name] [expr: \$method_expr]"
        methods: "Query,Search,Delete"
  connectionCheckIntervalSeconds: 120 # the interval time(in seconds) for connection manager to scan inactive client info
  connectionClientInfoTTLSeconds: 86400 # inactive client info TTL duration, in seconds
  maxConnectionNum: 10000 # the max client info numbers that proxy should manage, avoid too many client infos
  gracefulStopTimeout: 30 # seconds. force stop node without graceful stop
  slowQuerySpanInSeconds: 5 # query whose executed time exceeds the \`slowQuerySpanInSeconds\` can be considered slow, in seconds.
  http:
    enabled: true # Whether to enable the http server
    debug_mode: false # Whether to enable http server debug mode
    port:  # high-level restful api
    acceptTypeAllowInt64: true # high-level restful api, whether http client can deal with int64
    enablePprof: true # Whether to enable pprof middleware on the metrics port
  ip:  # if not specified, use the first unicastable address
  port: 19530
  internalPort: 19529
  grpc:
    serverMaxSendSize: 268435456
    serverMaxRecvSize: 67108864
    clientMaxSendSize: 268435456
    clientMaxRecvSize: 67108864

# Related configuration of queryCoord, used to manage topology and load balancing for the query nodes, and handoff from growing segments to sealed segments.
queryCoord:
  taskMergeCap: 1
  taskExecutionCap: 256
  autoHandoff: true # Enable auto handoff
  autoBalance: true # Enable auto balance
  autoBalanceChannel: true # Enable auto balance channel
  balancer: ScoreBasedBalancer # auto balancer used for segments on queryNodes
  globalRowCountFactor: 0.1 # the weight used when balancing segments among queryNodes
  scoreUnbalanceTolerationFactor: 0.05 # the least value for unbalanced extent between from and to nodes when doing balance
  reverseUnBalanceTolerationFactor: 1.3 # the largest value for unbalanced extent between from and to nodes after doing balance
  overloadedMemoryThresholdPercentage: 90 # The threshold percentage that memory overload
  balanceIntervalSeconds: 60
  memoryUsageMaxDifferencePercentage: 30
  rowCountFactor: 0.4 # the row count weight used when balancing segments among queryNodes
  segmentCountFactor: 0.4 # the segment count weight used when balancing segments among queryNodes
  globalSegmentCountFactor: 0.1 # the segment count weight used when balancing segments among queryNodes
  segmentCountMaxSteps: 50 # segment count based plan generator max steps
  rowCountMaxSteps: 50 # segment count based plan generator max steps
  randomMaxSteps: 10 # segment count based plan generator max steps
  growingRowCountWeight: 4 # the memory weight of growing segment row count
  balanceCostThreshold: 0.001 # the threshold of balance cost, if the difference of cluster's cost after executing the balance plan is less than this value, the plan will not be executed
  checkSegmentInterval: 1000
  checkChannelInterval: 1000
  checkBalanceInterval: 10000
  checkIndexInterval: 10000
  channelTaskTimeout: 60000 # 1 minute
  segmentTaskTimeout: 120000 # 2 minute
  distPullInterval: 500
  heartbeatAvailableInterval: 10000 # 10s, Only QueryNodes which fetched heartbeats within the duration are available
  loadTimeoutSeconds: 600
  distRequestTimeout: 5000 # the request timeout for querycoord fetching data distribution from querynodes, in milliseconds
  heatbeatWarningLag: 5000 # the lag value for querycoord report warning when last heatbeat is too old, in milliseconds
  checkHandoffInterval: 5000
  enableActiveStandby: false
  checkInterval: 1000
  checkHealthInterval: 3000 # 3s, the interval when query coord try to check health of query node
  checkHealthRPCTimeout: 2000 # 100ms, the timeout of check health rpc to query node
  brokerTimeout: 5000 # 5000ms, querycoord broker rpc timeout
  collectionRecoverTimes: 3 # if collection recover times reach the limit during loading state, release it
  observerTaskParallel: 16 # the parallel observer dispatcher task number
  checkAutoBalanceConfigInterval: 10 # the interval of check auto balance config
  checkNodeSessionInterval: 60 # the interval(in seconds) of check querynode cluster session
  gracefulStopTimeout: 5 # seconds. force stop node without graceful stop
  enableStoppingBalance: true # whether enable stopping balance
  cleanExcludeSegmentInterval: 60 # the time duration of clean pipeline exclude segment which used for filter invalid data, in seconds
  ip:  # if not specified, use the first unicastable address
  port: 19531
  grpc:
    serverMaxSendSize: 536870912
    serverMaxRecvSize: 268435456
    clientMaxSendSize: 268435456
    clientMaxRecvSize: 536870912

# Related configuration of queryNode, used to run hybrid search between vector and scalar data.
queryNode:
  stats:
    publishInterval: 1000 # Interval for querynode to report node information (milliseconds)
  segcore:
    knowhereThreadPoolNumRatio: 4 # The number of threads in knowhere's thread pool. If disk is enabled, the pool size will multiply with knowhereThreadPoolNumRatio([1, 32]).
    chunkRows: 128 # The number of vectors in a chunk.
    interimIndex:
      enableIndex: true # Enable segment build with index to accelerate vector search when segment is in growing or binlog.
      nlist: 128 # temp index nlist, recommend to set sqrt(chunkRows), must smaller than chunkRows/8
      nprobe: 16 # nprobe to search small index, based on your accuracy requirement, must smaller than nlist
      memExpansionRate: 1.15 # extra memory needed by building interim index
      buildParallelRate: 0.5 # the ratio of building interim index parallel matched with cpu num
  loadMemoryUsageFactor: 1 # The multiply factor of calculating the memory usage while loading segments
  enableDisk: false # enable querynode load disk index, and search on disk index
  maxDiskUsagePercentage: 95
  cache:
    enabled: true
    memoryLimit: 2147483648 # 2 GB, 2 * 1024 *1024 *1024
    readAheadPolicy: willneed # The read ahead policy of chunk cache, options: \`normal, random, sequential, willneed, dontneed\`
    # options: async, sync, off. 
    # Specifies the necessity for warming up the chunk cache. 
    # 1. If set to "sync" or "async," the original vector data will be synchronously/asynchronously loaded into the 
    # chunk cache during the load process. This approach has the potential to substantially reduce query/search latency
    # for a specific duration post-load, albeit accompanied by a concurrent increase in disk usage;
    # 2. If set to "off," original vector data will only be loaded into the chunk cache during search/query.
    warmup: async
  mmap:
    mmapEnabled: false # Enable mmap for loading data
  mmapEnabled: false # Enable mmap for loading data
  lazyloadEnabled: false # Enable lazyload for loading data
  grouping:
    enabled: true
    maxNQ: 1000
    topKMergeRatio: 20
  scheduler:
    receiveChanSize: 10240
    unsolvedQueueSize: 10240
    # maxReadConcurrentRatio is the concurrency ratio of read task (search task and query task).
    # Max read concurrency would be the value of hardware.GetCPUNum * maxReadConcurrentRatio.
    # It defaults to 2.0, which means max read concurrency would be the value of hardware.GetCPUNum * 2.
    # Max read concurrency must greater than or equal to 1, and less than or equal to hardware.GetCPUNum * 100.
    # (0, 100]
    maxReadConcurrentRatio: 1
    cpuRatio: 10 # ratio used to estimate read task cpu usage.
    maxTimestampLag: 86400
    scheduleReadPolicy:
      # fifo: A FIFO queue support the schedule.
      # user-task-polling:
      # 	The user's tasks will be polled one by one and scheduled.
      # 	Scheduling is fair on task granularity.
      # 	The policy is based on the username for authentication.
      # 	And an empty username is considered the same user.
      # 	When there are no multi-users, the policy decay into FIFO"
      name: fifo
      taskQueueExpire: 60 # Control how long (many seconds) that queue retains since queue is empty
      enableCrossUserGrouping: false # Enable Cross user grouping when using user-task-polling policy. (Disable it if user's task can not merge each other)
      maxPendingTaskPerUser: 1024 # Max pending task per user in scheduler
  dataSync:
    flowGraph:
      maxQueueLength: 16 # Maximum length of task queue in flowgraph
      maxParallelism: 1024 # Maximum number of tasks executed in parallel in the flowgraph
  enableSegmentPrune: false # use partition prune function on shard delegator
  ip:  # if not specified, use the first unicastable address
  port: 21123
  grpc:
    serverMaxSendSize: 536870912
    serverMaxRecvSize: 268435456
    clientMaxSendSize: 268435456
    clientMaxRecvSize: 536870912

indexCoord:
  bindIndexNodeMode:
    enable: false
    address: localhost:22930
    withCred: false
    nodeID: 0
  segment:
    minSegmentNumRowsToEnableIndex: 1024 # It's a threshold. When the segment num rows is less than this value, the segment will not be indexed

indexNode:
  scheduler:
    buildParallel: 1
  enableDisk: true # enable index node build disk vector index
  maxDiskUsagePercentage: 95
  ip:  # if not specified, use the first unicastable address
  port: 21121
  grpc:
    serverMaxSendSize: 536870912
    serverMaxRecvSize: 268435456
    clientMaxSendSize: 268435456
    clientMaxRecvSize: 536870912

dataCoord:
  channel:
    watchTimeoutInterval: 300 # Timeout on watching channels (in seconds). Datanode tickler update watch progress will reset timeout timer.
    balanceSilentDuration: 300 # The duration after which the channel manager start background channel balancing
    balanceInterval: 360 # The interval with which the channel manager check dml channel balance status
    checkInterval: 10 # The interval in seconds with which the channel manager advances channel states
    notifyChannelOperationTimeout: 5 # Timeout notifing channel operations (in seconds).
  segment:
    maxSize: 1024 # Maximum size of a segment in MB
    diskSegmentMaxSize: 2048 # Maximun size of a segment in MB for collection which has Disk index
    sealProportion: 0.12
    assignmentExpiration: 2000 # The time of the assignment expiration in ms
    allocLatestExpireAttempt: 200 # The time attempting to alloc latest lastExpire from rootCoord after restart
    maxLife: 86400 # The max lifetime of segment in seconds, 24*60*60
    # If a segment didn't accept dml records in maxIdleTime and the size of segment is greater than
    # minSizeFromIdleToSealed, Milvus will automatically seal it.
    # The max idle time of segment in seconds, 10*60.
    maxIdleTime: 600
    minSizeFromIdleToSealed: 16 # The min size in MB of segment which can be idle from sealed.
    # The max number of binlog file for one segment, the segment will be sealed if
    # the number of binlog file reaches to max value.
    maxBinlogFileNumber: 32
    smallProportion: 0.5 # The segment is considered as "small segment" when its # of rows is smaller than
    # (smallProportion * segment max # of rows).
    # A compaction will happen on small segments if the segment after compaction will have
    compactableProportion: 0.85
    # over (compactableProportion * segment max # of rows) rows.
    # MUST BE GREATER THAN OR EQUAL TO <smallProportion>!!!
    # During compaction, the size of segment # of rows is able to exceed segment max # of rows by (expansionRate-1) * 100%. 
    expansionRate: 1.25
  autoUpgradeSegmentIndex: false # whether auto upgrade segment index to index engine's version
  enableCompaction: true # Enable data segment compaction
  compaction:
    enableAutoCompaction: true
    indexBasedCompaction: true
    rpcTimeout: 10
    maxParallelTaskNum: 10
    workerMaxParallelTaskNum: 2
    levelzero:
      forceTrigger:
        minSize: 8388608 # The minmum size in bytes to force trigger a LevelZero Compaction, default as 8MB
        maxSize: 67108864 # The maxmum size in bytes to force trigger a LevelZero Compaction, default as 64MB
        deltalogMinNum: 10 # The minimum number of deltalog files to force trigger a LevelZero Compaction
        deltalogMaxNum: 30 # The maxmum number of deltalog files to force trigger a LevelZero Compaction, default as 30
  enableGarbageCollection: true
  gc:
    interval: 3600 # gc interval in seconds
    missingTolerance: 86400 # file meta missing tolerance duration in seconds, default to 24hr(1d)
    dropTolerance: 10800 # file belongs to dropped entity tolerance duration in seconds. 3600
    removeConcurrent: 32 # number of concurrent goroutines to remove dropped s3 objects
    scanInterval: 168 # garbage collection scan residue interval in hours
  enableActiveStandby: false
  brokerTimeout: 5000 # 5000ms, dataCoord broker rpc timeout
  autoBalance: true # Enable auto balance
  checkAutoBalanceConfigInterval: 10 # the interval of check auto balance config
  import:
    filesPerPreImportTask: 2 # The maximum number of files allowed per pre-import task.
    taskRetention: 10800 # The retention period in seconds for tasks in the Completed or Failed state.
    maxSizeInMBPerImportTask: 6144 # To prevent generating of small segments, we will re-group imported files. This parameter represents the sum of file sizes in each group (each ImportTask).
    scheduleInterval: 2 # The interval for scheduling import, measured in seconds.
    checkIntervalHigh: 2 # The interval for checking import, measured in seconds, is set to a high frequency for the import checker.
    checkIntervalLow: 120 # The interval for checking import, measured in seconds, is set to a low frequency for the import checker.
    maxImportFileNumPerReq: 1024 # The maximum number of files allowed per single import request.
    waitForIndex: true # Indicates whether the import operation waits for the completion of index building.
  gracefulStopTimeout: 5 # seconds. force stop node without graceful stop
  ip:  # if not specified, use the first unicastable address
  port: 13333
  grpc:
    serverMaxSendSize: 536870912
    serverMaxRecvSize: 268435456
    clientMaxSendSize: 268435456
    clientMaxRecvSize: 536870912

dataNode:
  dataSync:
    flowGraph:
      maxQueueLength: 16 # Maximum length of task queue in flowgraph
      maxParallelism: 1024 # Maximum number of tasks executed in parallel in the flowgraph
    maxParallelSyncMgrTasks: 256 # The max concurrent sync task number of datanode sync mgr globally
    skipMode:
      enable: true # Support skip some timetick message to reduce CPU usage
      skipNum: 4 # Consume one for every n records skipped
      coldTime: 60 # Turn on skip mode after there are only timetick msg for x seconds
  segment:
    insertBufSize: 16777216 # Max buffer size to flush for a single segment.
    deleteBufBytes: 67108864 # Max buffer size in bytes to flush del for a single channel, default as 16MB
    syncPeriod: 600 # The period to sync segments if buffer is not empty.
  memory:
    forceSyncEnable: true # Set true to force sync if memory usage is too high
    forceSyncSegmentNum: 1 # number of segments to sync, segments with top largest buffer will be synced.
    checkInterval: 3000 # the interal to check datanode memory usage, in milliseconds
    forceSyncWatermark: 0.5 # memory watermark for standalone, upon reaching this watermark, segments will be synced.
  timetick:
    byRPC: true
    interval: 500
  channel:
    # specify the size of global work pool of all channels
    # if this parameter <= 0, will set it as the maximum number of CPUs that can be executing
    # suggest to set it bigger on large collection numbers to avoid blocking
    workPoolSize: -1
    # specify the size of global work pool for channel checkpoint updating
    # if this parameter <= 0, will set it as 10
    updateChannelCheckpointMaxParallel: 10
    updateChannelCheckpointInterval: 60 # the interval duration(in seconds) for datanode to update channel checkpoint of each channel
    updateChannelCheckpointRPCTimeout: 20 # timeout in seconds for UpdateChannelCheckpoint RPC call
    maxChannelCheckpointsPerPRC: 128 # The maximum number of channel checkpoints per UpdateChannelCheckpoint RPC.
    channelCheckpointUpdateTickInSeconds: 10 # The frequency, in seconds, at which the channel checkpoint updater executes updates.
  import:
    maxConcurrentTaskNum: 16 # The maximum number of import/pre-import tasks allowed to run concurrently on a datanode.
    maxImportFileSizeInGB: 16 # The maximum file size (in GB) for an import file, where an import file refers to either a Row-Based file or a set of Column-Based files.
    readBufferSizeInMB: 16 # The data block size (in MB) read from chunk manager by the datanode during import.
  compaction:
    levelZeroBatchMemoryRatio: 0.05 # The minimal memory ratio of free memory for level zero compaction executing in batch mode
  gracefulStopTimeout: 1800 # seconds. force stop node without graceful stop
  ip:  # if not specified, use the first unicastable address
  port: 21124
  grpc:
    serverMaxSendSize: 536870912
    serverMaxRecvSize: 268435456
    clientMaxSendSize: 268435456
    clientMaxRecvSize: 536870912

# Configures the system log output.
log:
  level: info # Only supports debug, info, warn, error, panic, or fatal. Default 'info'.
  file:
    rootPath:  # root dir path to put logs, default "" means no log file will print. please adjust in embedded Milvus: /tmp/milvus/logs
    maxSize: 300 # MB
    maxAge: 10 # Maximum time for log retention in day.
    maxBackups: 20
  format: text # text or json
  stdout: true # Stdout enable or not

grpc:
  log:
    level: WARNING
  serverMaxSendSize: 536870912
  serverMaxRecvSize: 268435456
  gracefulStopTimeout: 10 # second, time to wait graceful stop finish
  client:
    compressionEnabled: false
    dialTimeout: 200
    keepAliveTime: 10000
    keepAliveTimeout: 20000
    maxMaxAttempts: 10
    initialBackoff: 0.2
    maxBackoff: 10
    minResetInterval: 1000
    maxCancelError: 32
    minSessionCheckInterval: 200
  clientMaxSendSize: 268435456
  clientMaxRecvSize: 536870912

# Configure the proxy tls enable.
tls:
  serverPemPath: configs/cert/server.pem
  serverKeyPath: configs/cert/server.key
  caPemPath: configs/cert/ca.pem

common:
  chanNamePrefix:
    cluster: by-dev
    rootCoordTimeTick: rootcoord-timetick
    rootCoordStatistics: rootcoord-statistics
    rootCoordDml: rootcoord-dml
    replicateMsg: replicate-msg
    queryTimeTick: queryTimeTick
    dataCoordTimeTick: datacoord-timetick-channel
    dataCoordSegmentInfo: segment-info-channel
  subNamePrefix:
    dataCoordSubNamePrefix: dataCoord
    dataNodeSubNamePrefix: dataNode
  defaultPartitionName: _default # default partition name for a collection
  defaultIndexName: _default_idx # default index name
  entityExpiration: -1 # Entity expiration in seconds, CAUTION -1 means never expire
  indexSliceSize: 16 # MB
  threadCoreCoefficient:
    highPriority: 10 # This parameter specify how many times the number of threads is the number of cores in high priority pool
    middlePriority: 5 # This parameter specify how many times the number of threads is the number of cores in middle priority pool
    lowPriority: 1 # This parameter specify how many times the number of threads is the number of cores in low priority pool
  buildIndexThreadPoolRatio: 0.75
  DiskIndex:
    MaxDegree: 56
    SearchListSize: 100
    PQCodeBudgetGBRatio: 0.125
    BuildNumThreadsRatio: 1
    SearchCacheBudgetGBRatio: 0.1
    LoadNumThreadRatio: 8
    BeamWidthRatio: 4
  gracefulTime: 5000 # milliseconds. it represents the interval (in ms) by which the request arrival time needs to be subtracted in the case of Bounded Consistency.
  gracefulStopTimeout: 1800 # seconds. it will force quit the server if the graceful stop process is not completed during this time.
  storageType: remote # please adjust in embedded Milvus: local, available values are [local, remote, opendal], value minio is deprecated, use remote instead
  # Default value: auto
  # Valid values: [auto, avx512, avx2, avx, sse4_2]
  # This configuration is only used by querynode and indexnode, it selects CPU instruction set for Searching and Index-building.
  simdType: auto
  security:
    authorizationEnabled: false
    # The superusers will ignore some system check processes,
    # like the old password verification when updating the credential
    superUsers: 
    tlsMode: 0
  session:
    ttl: 30 # ttl value when session granting a lease to register service
    retryTimes: 30 # retry times when session sending etcd requests
  locks:
    metrics:
      enable: false # whether gather statistics for metrics locks
    threshold:
      info: 500 # minimum milliseconds for printing durations in info level
      warn: 1000 # minimum milliseconds for printing durations in warn level
  storage:
    scheme: s3
    enablev2: false
  ttMsgEnabled: true # Whether the instance disable sending ts messages
  traceLogMode: 0 # trace request info
  bloomFilterSize: 100000 # bloom filter initial size
  maxBloomFalsePositive: 0.05 # max false positive rate for bloom filter

# QuotaConfig, configurations of Milvus quota and limits.
# By default, we enable:
#   1. TT protection;
#   2. Memory protection.
#   3. Disk quota protection.
# You can enable:
#   1. DML throughput limitation;
#   2. DDL, DQL qps/rps limitation;
#   3. DQL Queue length/latency protection;
#   4. DQL result rate protection;
# If necessary, you can also manually force to deny RW requests.
quotaAndLimits:
  enabled: true # \`true\` to enable quota and limits, \`false\` to disable.
  # quotaCenterCollectInterval is the time interval that quotaCenter
  # collects metrics from Proxies, Query cluster and Data cluster.
  # seconds, (0 ~ 65536)
  quotaCenterCollectInterval: 3
  ddl:
    enabled: false
    collectionRate: -1 # qps, default no limit, rate for CreateCollection, DropCollection, LoadCollection, ReleaseCollection
    partitionRate: -1 # qps, default no limit, rate for CreatePartition, DropPartition, LoadPartition, ReleasePartition
    db:
      collectionRate: -1 # qps of db level , default no limit, rate for CreateCollection, DropCollection, LoadCollection, ReleaseCollection
      partitionRate: -1 # qps of db level, default no limit, rate for CreatePartition, DropPartition, LoadPartition, ReleasePartition
  indexRate:
    enabled: false
    max: -1 # qps, default no limit, rate for CreateIndex, DropIndex
    db:
      max: -1 # qps of db level, default no limit, rate for CreateIndex, DropIndex
  flushRate:
    enabled: false
    max: -1 # qps, default no limit, rate for flush
    collection:
      max: -1 # qps, default no limit, rate for flush at collection level.
    db:
      max: -1 # qps of db level, default no limit, rate for flush
  compactionRate:
    enabled: false
    max: -1 # qps, default no limit, rate for manualCompaction
    db:
      max: -1 # qps of db level, default no limit, rate for manualCompaction
  dml:
    # dml limit rates, default no limit.
    # The maximum rate will not be greater than max.
    enabled: false
    insertRate:
      max: -1 # MB/s, default no limit
      db:
        max: -1 # MB/s, default no limit
      collection:
        max: -1 # MB/s, default no limit
      partition:
        max: -1 # MB/s, default no limit
    upsertRate:
      max: -1 # MB/s, default no limit
      db:
        max: -1 # MB/s, default no limit
      collection:
        max: -1 # MB/s, default no limit
      partition:
        max: -1 # MB/s, default no limit
    deleteRate:
      max: -1 # MB/s, default no limit
      db:
        max: -1 # MB/s, default no limit
      collection:
        max: -1 # MB/s, default no limit
      partition:
        max: -1 # MB/s, default no limit
    bulkLoadRate:
      max: -1 # MB/s, default no limit, not support yet. TODO: limit bulkLoad rate
      db:
        max: -1 # MB/s, default no limit, not support yet. TODO: limit db bulkLoad rate
      collection:
        max: -1 # MB/s, default no limit, not support yet. TODO: limit collection bulkLoad rate
      partition:
        max: -1 # MB/s, default no limit, not support yet. TODO: limit partition bulkLoad rate
  dql:
    # dql limit rates, default no limit.
    # The maximum rate will not be greater than max.
    enabled: false
    searchRate:
      max: -1 # vps (vectors per second), default no limit
      db:
        max: -1 # vps (vectors per second), default no limit
      collection:
        max: -1 # vps (vectors per second), default no limit
      partition:
        max: -1 # vps (vectors per second), default no limit
    queryRate:
      max: -1 # qps, default no limit
      db:
        max: -1 # qps, default no limit
      collection:
        max: -1 # qps, default no limit
      partition:
        max: -1 # qps, default no limit
  limits:
    maxCollectionNum: 65536
    maxCollectionNumPerDB: 65536
    maxResourceGroupNumOfQueryNode: 1024 # maximum number of resource groups of query nodes
  limitWriting:
    # forceDeny false means dml requests are allowed (except for some
    # specific conditions, such as memory of nodes to water marker), true means always reject all dml requests.
    forceDeny: false
    ttProtection:
      enabled: false
      # maxTimeTickDelay indicates the backpressure for DML Operations.
      # DML rates would be reduced according to the ratio of time tick delay to maxTimeTickDelay,
      # if time tick delay is greater than maxTimeTickDelay, all DML requests would be rejected.
      # seconds
      maxTimeTickDelay: 300
    memProtection:
      # When memory usage > memoryHighWaterLevel, all dml requests would be rejected;
      # When memoryLowWaterLevel < memory usage < memoryHighWaterLevel, reduce the dml rate;
      # When memory usage < memoryLowWaterLevel, no action.
      enabled: true
      dataNodeMemoryLowWaterLevel: 0.85 # (0, 1], memoryLowWaterLevel in DataNodes
      dataNodeMemoryHighWaterLevel: 0.95 # (0, 1], memoryHighWaterLevel in DataNodes
      queryNodeMemoryLowWaterLevel: 0.85 # (0, 1], memoryLowWaterLevel in QueryNodes
      queryNodeMemoryHighWaterLevel: 0.95 # (0, 1], memoryHighWaterLevel in QueryNodes
    growingSegmentsSizeProtection:
      # No action will be taken if the growing segments size is less than the low watermark.
      # When the growing segments size exceeds the low watermark, the dml rate will be reduced,
      # but the rate will not be lower than minRateRatio * dmlRate.
      enabled: false
      minRateRatio: 0.5
      lowWaterLevel: 0.2
      highWaterLevel: 0.4
    diskProtection:
      enabled: true # When the total file size of object storage is greater than \`diskQuota\`, all dml requests would be rejected;
      diskQuota: -1 # MB, (0, +inf), default no limit
      diskQuotaPerDB: -1 # MB, (0, +inf), default no limit
      diskQuotaPerCollection: -1 # MB, (0, +inf), default no limit
      diskQuotaPerPartition: -1 # MB, (0, +inf), default no limit
  limitReading:
    # forceDeny false means dql requests are allowed (except for some
    # specific conditions, such as collection has been dropped), true means always reject all dql requests.
    forceDeny: false
    queueProtection:
      enabled: false
      # nqInQueueThreshold indicated that the system was under backpressure for Search/Query path.
      # If NQ in any QueryNode's queue is greater than nqInQueueThreshold, search&query rates would gradually cool off
      # until the NQ in queue no longer exceeds nqInQueueThreshold. We think of the NQ of query request as 1.
      # int, default no limit
      nqInQueueThreshold: -1
      # queueLatencyThreshold indicated that the system was under backpressure for Search/Query path.
      # If dql latency of queuing is greater than queueLatencyThreshold, search&query rates would gradually cool off
      # until the latency of queuing no longer exceeds queueLatencyThreshold.
      # The latency here refers to the averaged latency over a period of time.
      # milliseconds, default no limit
      queueLatencyThreshold: -1
    resultProtection:
      enabled: false
      # maxReadResultRate indicated that the system was under backpressure for Search/Query path.
      # If dql result rate is greater than maxReadResultRate, search&query rates would gradually cool off
      # until the read result rate no longer exceeds maxReadResultRate.
      # MB/s, default no limit
      maxReadResultRate: -1
      maxReadResultRatePerDB: -1
      maxReadResultRatePerCollection: -1
    # colOffSpeed is the speed of search&query rates cool off.
    # (0, 1]
    coolOffSpeed: 0.9

trace:
  # trace exporter type, default is stdout,
  # optional values: ['stdout', 'jaeger', 'otlp']
  exporter: stdout
  # fraction of traceID based sampler,
  # optional values: [0, 1]
  # Fractions >= 1 will always sample. Fractions < 0 are treated as zero.
  sampleFraction: 0
  jaeger:
    url:  # when exporter is jaeger should set the jaeger's URL
  otlp:
    endpoint:  # example: "127.0.0.1:4318"
    secure: true

#when using GPU indexing, Milvus will utilize a memory pool to avoid frequent memory allocation and deallocation.
#here, you can set the size of the memory occupied by the memory pool, with the unit being MB.
#note that there is a possibility of Milvus crashing when the actual memory demand exceeds the value set by maxMemSize.
#if initMemSize and MaxMemSize both set zero,
#milvus will automatically initialize half of the available GPU memory,
#maxMemSize will the whole available GPU memory.
gpu:
  initMemSize:  # Gpu Memory Pool init size
  maxMemSize:  # Gpu Memory Pool Max size
EOF

