# ScAI 管理员手册

## 1. 目的与读者

本手册面向负责部署和运维单个 ScAI 实例、管理账户与模型凭据、保护持久化数据、监控服务健康状况以及支持程序化使用的管理员。分析用户应改为阅读[用户指南](user-guide-cn.md)。Python 包的详细信息也可参阅[客户端 README](../clients/python/README.md)，但若该文档与当前客户端源码存在差异，应以当前客户端源码为准。

示例使用 `<API_KEY>`、`<SESSION_ID>` 和 `<ARCHIVE_PATH>` 等明确标记的值。除非另有说明，请从仓库根目录运行命令。

## 2. 架构与要求

ScAI 在生产模式下由同一个 FastAPI/Uvicorn 进程在端口 `3000` 提供已构建的 Vite/React 应用、REST API 和 WebSocket。开发模式仍在 `3000` 运行 Vite，并将请求代理到 `8000` 上的 FastAPI/Uvicorn。两种模式都使用本地分析内核、外部 OpenAI 兼容模型服务，以及基于磁盘的应用与会话存储。

支持以下部署方式：

- Docker：Node.js 20.19 构建阶段编译前端；Python 3.11 运行时不包含 Node.js/npm，只公开应用端口 `3000`。
- 直接启动：需要 Linux 环境及与 GNU 兼容的 `find`、`grep`、`head`、`cut`、`tr`、`cmp` 和 `cp`；用于创建虚拟环境的 `python3.12`；`python3`；以及 `pgrep`、`pkill`、`lsof` 和 `kill`。开发模式以及生产前端产物缺失或过期时，还需要 Node.js 20.19+ 或 22.12+ 以及 npm/npx。这些脚本并非按仅兼容可移植 POSIX 工具的方式设计。

请为上传的数据集、生成的产物、日志、临时上传文件和依赖项预留存储空间。主机必须能够访问已配置的模型端点，并在启动期间访问 Python 软件包注册表。Docker 启动会安装 Python requirements，因此首次启动可能明显较慢；前端依赖只在构建镜像时解析。

### 内存容量规划

实例的内存几乎全部消耗在分析内核上，而不是后端。**每个处于分析中的会话各自持有一个
内核进程**，因此主机若要并发运行 N 个分析，所需内存为下表单次分析数值的 N 倍，再加
上固定的平台开销。

单细胞流程内存开销大有其结构性原因：stage-1 运行会同时持有各样本矩阵、它们经 QC 过
滤后的副本，以及一个带 counts 层、lognorm 层和 `.raw` 的合并对象。在做任何稠密变换之
前，同一份表达数据就已存在六到十份拷贝。

**基准实测。** 在参考部署上，一个包含 2 个样本、19,128 个细胞、22,804 个基因的会话
（`.h5ad` 落盘 392 MB），其内核常驻内存达到 **6.4 GiB**，同时后端进程占用 0.96 GiB。
下面的估算模型即拟合于该实测点：

```
内核峰值 (GiB) ≈ 0.4 + (细胞数 × 330 KB) + (样本数 × 50 MB)
平台开销      ≈ 1.6 GiB   （后端、前端、操作系统余量）
```

推荐值已在模型峰值之上包含 **30% 的安全冗余**。

| 样本数 | 细胞总数 |  内核峰值 | **推荐配置（不启用快照）** | **推荐配置（启用快照）** |
| -----: | -------: | --------: | -------------------------------: | -----------------------------: |
|      2 |   10,000 |   3.6 GiB |                  **7 GiB** |                **9 GiB** |
|      2 |   20,000 |   6.8 GiB |                 **11 GiB** |               **14 GiB** |
|      4 |   40,000 |  13.2 GiB |                 **20 GiB** |               **26 GiB** |
|      6 |   60,000 |  19.6 GiB |                 **28 GiB** |               **37 GiB** |
|      8 |  100,000 |  32.3 GiB |                 **44 GiB** |               **59 GiB** |
|     12 |  200,000 |  63.9 GiB |                 **86 GiB** |              **115 GiB** |
|     20 |  500,000 | 158.7 GiB |                **209 GiB** |              **281 GiB** |
|    100 | 1,000,000 | 320.0 GiB |                **418 GiB** |              **564 GiB** |

两个推荐列均为**单次并发分析**的数值。快照列的依据见[第 8 节](#快照的内存影响)；若不
启用快照，请使用中间那一列。

**本表的假设与适用边界。**

- 假设为全转录组数据，基因数 15,000–35,000。基因数远少于此的 panel 数据按比例递减；
  每细胞检出深度很高的靶向实验则更高。
- 主导因素是细胞数。内存随非零元素个数增长，而非零元素随细胞数增长，与基因注释的
  大小无关。
- 模型仅拟合于**一个**实测部署。200,000、500,000 与 1,000,000 细胞三行属于外推而非
  实测，在按该规模采购硬件前请先行验证。
- 峰值是瞬时的。对完整矩阵做稠密变换会在表格数值之外额外消耗
  `细胞数 × 基因数 × 4 字节` —— 对基准数据集为 1.6 GiB，且该项随总基因数而非检出基因
  数增长。把此类操作限制在高变基因上的分析远低于此值。

100 样本这一行假设每个样本约保留 10,000 个细胞。若总细胞数相同，换成其他样本数时矩阵
成本近似，样本数主要增加相对较小的逐样本 QC 与元数据开销。该行用于容量规划，不表示所有
分析方法都已经在 100 万细胞上完成验证。

#### 运行时持久化预算

常规运行时持久化使用一套独立于 Snapshot V2 的资源门禁。默认情况下，内存与文件大小上限
为**自动计算**：优先使用 Docker cgroup 内存限额；没有限额时使用宿主机内存：

| 预算 | 自动上限 |
|---|---:|
| 最大单对象 | 检测容量的 25% |
| 存活 AnnData 总量 | 45% |
| 分析内核 RSS | 60% |
| 压缩运行时文件 | 30% |

这些是逐会话的持久化上限，不是全局调度器。共享实例必须按最大并发存活分析内核数量设置
容器容量或逐会话绝对上限。未限制内存的容器会让每个会话都看到整台主机容量，两个会话
各自都可能获得单独看似有效、但相加后超过主机的预算。应优先设置有限的容器内存，或使用
第 17 节环境变量表中的 `SCAI_RUNTIME_MAX_*` 设置绝对上限。

容量变量未设置和显式设置为 `auto` 都表示自动计算。正数表示管理员指定的 GiB 上限，最终
有效值取该数值和自动上限中较小者。例如：

```bash
docker run -d --name scai --memory=128g \
  -e SCAI_RUNTIME_MAX_RSS_GB=64 \
  -e SCAI_RUNTIME_MAX_SINGLE_OBJECT_GB=32 \
  -e SCAI_RUNTIME_MAX_ANNDATA_TOTAL_GB=48 \
  -e SCAI_RUNTIME_MAX_PKL_GB=16 \
  -e SCAI_RUNTIME_MAX_SAVE_SECONDS=900 \
  ... ghcr.io/minzhangcheng/scai:latest
```

可参考以下显式配置档位：

| 总细胞数 | RSS | 单对象 | AnnData 总量 | 运行时文件 | 保存超时 |
|---:|---:|---:|---:|---:|---:|
| 不超过 50,000 | 32 GiB | 16 GiB | 24 GiB | 8 GiB | 300 秒 |
| 50,000–200,000 | 64 GiB | 32 GiB | 48 GiB | 16 GiB | 900 秒 |
| 200,000–500,000 | 256 GiB | 128 GiB | 192 GiB | 64 GiB | 1,800 秒 |
| 约 1,000,000 | 512 GiB | 256 GiB | 384 GiB | 128 GiB | 3,600 秒 |

后两个档位属于外推，并假设每份内存配额内只运行一个大型分析。除非管理员需要设置更严格
的并发边界，否则应使用自动默认值。磁盘空间只能保存 checkpoint，不会驱逐存活 Python
对象或降低内核 RSS。常规持久化还可能需要将 backed AnnData 物化到内存，因此磁盘化输入
本身不等于 out-of-core 执行。

**不要在没有内存上限的情况下运行实例。** 失控的内核耗尽主机内存后，会招致内核 OOM
killer 终止任意进程，且可能是同一主机上的其他服务。请依据上表设置容器限额：

```bash
docker run -d --name scai --memory=32g ... ghcr.io/minzhangcheng/scai:latest
```

## 3. Docker 部署

### 拉取并运行

```bash
docker pull ghcr.io/minzhangcheng/scai:latest
docker run -d --name scai \
  -p 3000:3000 \
  -v "$(pwd)/scai-dep:/dep" \
  -v "$(pwd)/scai-data:/data" \
  --env-file <PROTECTED_ENV_FILE> \
  --restart unless-stopped \
  ghcr.io/minzhangcheng/scai:latest
```

**警告：**必须复用旧容器所使用的 `SCAI_JWT_SECRET` 和 `SCAI_CONFIG_SECRET`。以上示例为简洁起见使用受保护的环境变量文件；请限制其他用户读取该文件。使用不同密钥重建容器会使现有 JWT 失效，并可能导致已加密的模型凭据无法读取。

打开 `http://localhost:3000`。`/dep` 存储 Python 环境；`/data` 存储应用数据。前端产物已编译进镜像，并由 FastAPI 提供。入口程序会创建这两个挂载目录，将其所有内容的所有者递归设置为 UID `10001`（`scai`），然后放弃 root 权限。会话目录通常限制为 `0700` 模式。请确保主机目录可以安全地更改所有权；如果备份工具无法读取这些目录，请在主机上使用提升后的权限。不要仅为使备份命令成功而更改在线数据的所有权。

镜像仅公开 `3000`；UI、API 和 WebSocket 都使用该端口。运行时不存在可供额外发布的 `8000` 独立服务。

常用容器环境变量如下：

```bash
docker run -d --name scai \
  -p 3000:3000 \
  -v "$(pwd)/scai-dep:/dep" \
  -v "$(pwd)/scai-data:/data" \
  -e SCAI_MODE=prod \
  -e SCAI_JWT_SECRET='<STABLE_RANDOM_SECRET>' \
  -e SCAI_CONFIG_SECRET='<STABLE_RANDOM_SECRET>' \
  -e PIP_INDEX_URL='<PYTHON_PACKAGE_INDEX_URL>' \
  ghcr.io/minzhangcheng/scai:latest
```

`SCAI_MODE` 默认为 `prod`；镜像名称后可传入 `--prod`。生产运行时不包含 Node.js/npm，因此会明确拒绝 `--dev`；需要 Vite 热更新时，请在源码 checkout 中使用 `./start.sh --dev`。Docker 始终导出 `SCAI_DATA_DIR=/data`。如果设置了 `PIP_INDEX_URL`，依赖项安装会使用该值。

### 健康检查、日志、生命周期与移除

内置健康检查每 30 秒调用一次 `http://127.0.0.1:3000/api/health`，超时时间为 5 秒，启动宽限期为 60 秒。

```bash
docker ps --filter name=scai
docker inspect --format '{{.State.Health.Status}}' scai
curl -f http://localhost:3000/api/health
docker logs -f scai
docker stop scai
docker start scai
docker restart scai
```

容器输出还会追加到 `/data/logs/scai.log`。

### 更新与备份

更新前先备份数据卷，然后替换容器，同时保留两个主机挂载目录：

```bash
docker pull ghcr.io/minzhangcheng/scai:latest
docker stop scai
docker rm scai
docker run -d --name scai \
  -p 3000:3000 \
  -v "$(pwd)/scai-dep:/dep" \
  -v "$(pwd)/scai-data:/data" \
  --restart unless-stopped \
  ghcr.io/minzhangcheng/scai:latest
```

若要移除服务但保留数据，只需停止并删除容器。**警告：**删除 `scai-data` 会销毁账户、配置、会话、报告和上传文件。

## 4. 直接启动与关闭

安装 Python 3.12，克隆仓库，然后运行。当生产前端需要构建或使用开发模式时，还需安装 Node.js 20.19+ 或 22.12+ 以及 npm：

```bash
./start.sh
```

默认使用生产模式、主机 `127.0.0.1` 和单一应用端口 `3000`。在开发模式下，`--backport` 选择被代理的后端端口，默认为 `8000`。支持的选项严格限定如下：

```text
./start.sh [--dev | --prod] [--host HOST] [--port PORT] [--backport PORT]
./start.sh -h
./start.sh --help
```

用于网络访问和非默认端口的示例：

```bash
./start.sh --prod --host 0.0.0.0 --port 3000
```

相对于仓库目录 `<REPO>`，启动过程将依赖项存储在 `<REPO>/../dep`，将数据存储在 `<REPO>/../data`，并将进程元数据存储在 `<REPO>/.scai.pid`。它使用 `python3.12 -m venv` 创建环境、更新 pip、安装后端 requirements，然后在应用端口上启动 Uvicorn。如果任一前端构建输入比 `dist/index.html` 更新，脚本会先在 checkout 之外使用 `npm ci` 安装与 `package-lock.json` 严格一致的依赖图，然后重新构建。开发模式则同时记录 Uvicorn 和 Vite PID。

直接运行的环境控制项包括 `SCAI_DATA_DIR`（覆盖后端数据根目录）、`SCAI_JWT_SECRET`、`SCAI_CONFIG_SECRET`、`SCAI_ENABLE_REPRODUCIBILITY` 和 `SCAI_AUTO_FETCH_MARKERS=1`。最后一个选项会在可选标记数据库缺失时获取它们，失败不会阻止启动。设置 `SCAI_DATA_DIR` 后，`start.sh` 仍会创建并打印 `<REPO>/../data`，但后端实际读写覆盖后的路径；请监控和备份覆盖路径，而不是打印出的同级目录。

使用以下命令停止实例：

```bash
./stop.sh
```

**警告：**`stop.sh` 在读取 `.scai.pid` 后仍会使用范围较宽的进程名称后备匹配。它会释放 `start.sh` 记录的应用/后端端口（包括自定义值），因此可能终止占用其中某个端口的无关进程。在共享主机或多实例安装中，请先检查记录的 PID 和进程命令行，并优先只停止目标 PID：

```bash
lsof -i:<FRONTEND_PORT>
lsof -i:<BACKEND_PORT>
kill <PID>
```

服务运行期间不要删除有效的 `.scai.pid`。如果启动报告 PID 文件过期，请先确认其中的 PID 已不存在，然后仅删除该过期文件。

## 5. 构建与发布镜像

直接构建：

```bash
docker build -t ghcr.io/minzhangcheng/scai:latest .
```

发布辅助脚本始终同时构建所选标签和 `latest`：

```text
./build.sh [--version TAG] [--push] [-h | --help]
```

```bash
./build.sh --version 0.39.26
./build.sh --version 0.39.26 --push
```

如果未指定 `--version`，辅助脚本使用 `git describe --tags --abbrev=0` 返回的最新 Git 标签；如果不存在标签，则使用 `latest`。`--push` 会将两个标签推送到 `ghcr.io/minzhangcheng/scai`；使用前请先向该注册表进行 Docker 身份验证。该辅助脚本不接受注册表、镜像名称、平台或构建参数 flags。

## 6. 首次运行与用户管理

首次访问时，通过设置界面创建初始管理员。后端仅在初始化尚未记录前允许此操作。用户名和可选电子邮件地址必须唯一。

管理员可以创建已激活的普通用户、列出用户、重置密码、更改角色、更改账户状态及删除用户。自助注册会创建一个 `pending` 账户，在管理员激活之前无法登录。状态值为 `pending`、`readonly` 和 `active`；角色为 `admin` 和 `user`。管理员不能更改自己的角色/状态或删除自己，ScAI 也会阻止删除最后一名管理员或降低其权限。

**警告：**删除用户前，请停止该用户拥有的所有分析，并备份所需数据。删除用户会逐出缓存的会话，并移除持久化会话目录、旧版状态文件、API 密钥、加密的用户配置及数据库记录，但不会自行停止活动的工作流/任务对象。删除后，请确认该用户会话 ID 对应的工作流或 worker 均已不存在，并确认预期文件以及数据库/会话列表项已消失。

在管理界面中配置全局模型 API 密钥、模型 base URL 和可选代理。保存配置会将其持久化，但**不会**验证凭据。请另行执行显式验证操作。分析启动时会检查必需配置是否存在；提供商身份验证、所选模型可用性、配额和网络故障可能要到首次模型请求时才会显现。已激活用户可以存储按用户配置的模型设置；根据服务器配置解析规则，这些设置具有更高优先级。

每个已激活用户均可为自动化创建 API 密钥。密钥以 `scai_` 开头，归创建者所有，且完整值仅显示一次。请赋予其描述性标签，将其存入密钥管理器，并撤销不再使用的密钥。JWT 登录令牌在 72 小时后过期；API 密钥在撤销前始终有效。

## 7. 持久化数据与目录布局

Docker 使用 `/data`；直接启动通常使用同级 `data` 目录，除非 `SCAI_DATA_DIR` 将其覆盖。

| 数据根目录下的路径                                      | 用途                                                    |
| ------------------------------------------------------- | ------------------------------------------------------- |
| `app.db`                                              | SQLite 用户、API 密钥哈希、全局设置和加密的用户模型设置 |
| `.jwt_secret`                                         | 未提供环境密钥时自动生成的 JWT 签名密钥                 |
| `.config_secret`                                      | 未提供配置密钥时使用的本地加密材料                      |
| `sessions/<SESSION_ID>/`                              | 每个会话的权威元数据和文件                              |
| `sessions/<SESSION_ID>/session.json`                  | 会话元数据                                              |
| `sessions/<SESSION_ID>/uploads/`                      | 会话输入                                                |
| `sessions/<SESSION_ID>/workspace/`                    | 分析代码和图形                                          |
| `sessions/<SESSION_ID>/states/`                       | 工作流/checkpoint 文件                                  |
| `sessions/<SESSION_ID>/reports/`                      | 生成的报告                                              |
| `sessions/<SESSION_ID>/logs/`                         | 会话日志                                                |
| `uploads/`, `states/`, `workspace/`, `reports/` | 旧版顶层存储和兼容路径                                  |
| `logs/`                                               | 服务日志；Docker 写入`logs/scai.log`                  |
| `upload_temp/`                                        | 正在进行的上传数据                                      |

Docker `/dep` 仅需持久化 `/dep/venv`，前端产物已编译进镜像。直接启动使用 `<REPO>/../dep/venv` 和 `<REPO>/../dep/frontend/node_modules`。依赖项可重现并可重新构建，但持久化可缩短重启时间。

## 8. 快照（实验性 —— 开发中）

> **状态：开发中、不稳定、默认关闭。** 仅在可以承受丢失快照的实例上启用。磁盘布局
> 与 API 形态可能在没有迁移路径的情况下变更，因此某一版本创建的快照不保证能在后续
> 版本中恢复。端到端行为尚未在真实部署上完成验收 —— 自动化覆盖包含真实内核的捕获与
> 恢复，但没有记录过运维验收流程。该功能有意不写入用户手册；启用后快照页面会带有实
> 验性标识。请勿向最终用户宣告它是受支持的成果保护手段，也不要把它当作备份机制 ——
> 备份请参见第 9 节与第 13 节。

快照用于把会话回退到分析过程中较早的某个时点。一次快照会捕获会话的分析状态、
notebook、工作区文件、工作流 checkpoint，以及在分析内核在线时的 Jupyter 运行时（含
已加载的数据集）。恢复快照会回退会话；若该快照记录了可恢复的位置，分析还能从该处
继续。

### 启用方式

只有在后端启动时 `SCAI_ENABLE_SNAPSHOT_V2` 为真值，快照才会开启：

```bash
docker run -d --name scai \
  -e SCAI_ENABLE_SNAPSHOT_V2=true \
  ...
```

该变量在启动时读取一次，修改后需重启。未设置时，快照 API 按未启用响应，且不会写入
任何快照存储。`GET /api/config` 会以 `snapshot_v2_enabled` 返回解析后的取值。

### 快照的存储位置

位于数据根目录下每个会话的目录内：

```
sessions/<id>/v2/          快照根目录
├── objects/               内容寻址的数据块
├── committed/             每个快照一份带签名的 manifest
├── staging/               进行中的工作区
├── journals/              崩溃恢复所使用的事务记录
└── integrity/             签名材料
```

数据块采用内容寻址，因此未发生变化的数据集只保存一份、由引用它的所有快照共享。存储
增长取决于真正的新数据，而非快照数量 —— 但一旦数据确实发生变化，内核持有大型数据集
的会话每次捕获仍可能增加数十 GB。

快照属于会话目录的一部分，因此常规的数据根目录备份（第 9 节）会一并包含它们，删除
会话也会删除其快照。快照存储使用由本实例 `SCAI_CONFIG_SECRET` 派生的完整性密钥签名；
在不同密钥下恢复数据根目录会使其失效。

### 容量与清理

两道上限、对应的环境变量以及建议下限，请参见第 17 节的[快照容量](#快照容量)。日常
运维需要注意两点：

- **垃圾回收是事件驱动的，而非定时执行。** 不再被引用的对象所占空间是在删除快照时回
  收的（导入与崩溃恢复时也会触发），永远不会定时进行。持续累积快照的会话不会自行
  变小。
- **配额是按会话计的。** 最坏情况下的磁盘总占用为单会话配额乘以持有快照的会话数；剩
  余空间保留是唯一的卷级约束。请按预期的并发分析数量规划数据卷容量。

当某会话占用达到其配额的 90% 时，快照面板会提示用户清理。越过上限后，必需的捕获会
fail-closed 并把该分析停在一个需用户处理的门上，因此实际的失效表现是运行停滞，而不
是静默的数据丢失。

### 快照的内存影响

快照对磁盘是持久开销，对内存则是**瞬时开销**。捕获活动内核会在捕获期间抬高该内核的
内存峰值，原因是两种序列化器都会为正在写出的数据额外生成一份副本：

- **AnnData 对象（`h5ad` 序列化器）。** 捕获不能改动活动的分析对象，因此会先完整
  `.copy()` 一份再做清洗和写出。在该对象被捕获期间，它的表达矩阵、各 layer 和 `.raw`
  在内核中同时存在两份。
- **其余对象（`dill` 序列化器）。** 写出后，捕获会把文件读回并反序列化以验证数据可完
  整往返。这次回读是被验证值的第二份独立副本。

有两个特性限制了该开销的规模。对象是**逐个捕获**的，因此额外峰值取决于单个最大的被捕
获对象，而非所有对象之和。同时，捕获的目标仅为 AnnData 对象加上一组固定的规范结果变
量名，而不是整个命名空间。

规划时请按捕获期间**额外预留内核常驻内存的 35%** 计算。在参考部署上（内核 6.4 GiB），
主对象 `adata` —— 含 counts 层、lognorm 层、`.raw`、邻接图与降维结果 —— 约占 1.3 GiB，
其副本带来约 20% 的增量；35% 这一取值为 dill 回读，以及最大对象占比更高的会话留出了
余量。这一差额即[内存容量规划](#内存容量规划)中两个推荐列的区别。

这一点在运维上的意义在于：**捕获发生在工作流决定的时点，而不是主机空闲的时候。** 一个
恰好卡着内存上限跑得下来的分析，会偏偏在捕获时失败；而必需捕获失败会把会话停在门上。
若实例内存本就紧张，保持快照关闭是更划算的选择；在未按上述余量规划的主机上启用快照，
等于把一个磁盘容量特性变成内存容量问题。

#### 捕获前的内存预检

正因为上述开销是可预测的，捕获会在付出这笔开销之前先测量它。在复制任何东西之前，内核
会报告即将捕获的每个 AnnData 的真实字节数、内核自身的常驻内存，以及它实际可分配的内存
余量（取主机 `MemAvailable` 与 cgroup 限额中更紧的一个）。若最大对象的副本放不下，捕获
会在**分配之前**被拒绝，会话停在一个提供三个选择的门上：

| 选择 | 效果 |
|---|---|
| **仍然创建快照** | 绕过预检执行捕获，接受副本耗尽内存、内核被终止的风险。该豁免是一次性的：后续捕获会重新判断。 |
| **跳过快照并继续** | 不捕获直接继续，仅损失该边界上的恢复点。恢复操作之前不提供此选项，因为那次快照是唯一的安全点。 |
| **终止** | 停止整个分析。 |

此门上有意没有"重试"。预检是确定性的，重试会用同样的测量对同样的余量再判断一次，结果
必然相同 —— 门将永远无法解除。

两条适用范围需要注意：

- **快照关闭时，不运行这项快照复制预检。** 独立的常规运行时持久化门禁仍会在持久化边界
  按上述自动/管理员预算检查存活命名空间。
- **常规分析分配不会预先授权。** 一个步骤可以在下一次持久化边界之前分配内存；这段区间
  仍以容器内存限额作为最终防线。

余量可通过 `SCAI_SNAPSHOT_MEMORY_SAFETY_RATIO` 调整（默认 `0.8`，见第 17 节）。证据缺失
绝不会导致拒绝：若探针读不到，或余量无法确定，捕获会按未引入预检时的行为继续执行。

### 恢复行为

恢复过程是事务性的：会先创建一个 `pre_restore` 快照，因此恢复本身也可回滚，且每一步
都有 journal 记录。若进程在操作中途终止，下一次启动（以及下一次加载该会话）会将每个
未完成的 journal 推进到终态 —— 能回滚的就回滚，否则将该会话标记为 recovery-blocked，
而不是提供一个半应用的状态。被标记的会话在阻塞解除前拒绝运行；该状态及其原因会在会
话的快照状态中报告。

### 监控

- 快照活动记录在该会话的日志目录下（第 10 节）。
- 每个会话的存储占用可在快照面板中查看，也可通过
  `GET /api/session/{id}/snapshots-v2` 的 `store_status` 字段获取。
- 捕获失败会以门的形式呈现给用户，而不是静默报错；因此当某个分析停滞且状态中带有
  `snapshot_failure` 时，应首先检查磁盘容量。

## 9. 备份、升级、恢复与回滚

应优先执行静止状态下的文件级备份，使 SQLite 与会话文件保持一致：

1. 停止 ScAI。
2. 复制整个数据根目录（包括隐藏文件），保留所有权、模式、时间戳和链接。
3. 记录应用/镜像版本，并将稳定的密钥值单独保存在受保护的密钥存储中。
4. 重启 ScAI 并验证健康状况。

Docker bind mount 示例：

```bash
docker stop scai
sudo cp -a ./scai-data "./scai-data-backup-$(date +%Y%m%d-%H%M%S)"
sudo test -f "$(ls -dt ./scai-data-backup-* | head -1)/app.db"
docker start scai
```

**警告：**运行递归复制前，请确认可用磁盘空间和备份目标。`sudo cp -a` 会保留 UID `10001`、`0700` 等模式、时间戳和链接；不要为了备份访问而对在线卷执行 `chown` 或 `chmod`。请验证复制后的目录树、文件数、大小、隐藏密钥文件，并进行恢复测试，不要仅依赖示例中的基本 `app.db` 检查。不要将活动写入期间生成的副本视为事务一致。

升级时，请先备份，再针对现有数据根目录部署新代码/镜像，然后测试登录、配置可见性、会话列表和健康请求。在验收前保留旧镜像标签/代码修订。回滚时，停止新版本，并同时恢复与旧应用版本匹配的完整数据备份。除非在执行有文档记录的修复，否则绝不要混合来自不同备份时间的部分数据库和会话文件。

恢复过程：停止服务，将损坏的数据根目录移到一旁，将完整备份恢复到原路径，恢复所需所有权，提供相同密钥，启动匹配的应用版本，并在重新允许用户访问前完成验证。

可移植会话导出/导入用于在实例之间迁移单个会话。以 `include_runtime=true` 导出、并在导入时传 `X-Trust: true` 时会一并携带在线内核状态和工作流 checkpoint，导入后可继续分析；默认导出只携带内容，无法继续。两种方式都不能替代完整服务备份——权衡详见第 13 节。

## 10. 监控与日志

生产模式和 Docker 通过单一应用端口进行基本存活检查；只有直接启动的开发模式才在默认端口 `8000` 上单独暴露后端：

```bash
curl -f http://localhost:3000/api/health
curl -f http://127.0.0.1:8000/api/health  # 仅开发模式
```

对于 Docker，请结合健康状态和日志：

```bash
docker inspect --format '{{.State.Status}} {{.State.Health.Status}}' scai
docker logs --since 30m scai
docker stats scai
```

对于直接运行，请检查端口和进程：

```bash
lsof -i:3000
lsof -i:8000
ps -fp "$(awk -F= '/_PID=/{printf "%s,", $2}' .scai.pid | sed 's/,$//')"
```

请监控数据、依赖项及临时上传文件系统上的可用磁盘和 inode 容量；进程内存/CPU；容器重启次数；健康检查延迟；重复身份验证失败；模型提供商错误；以及 `logs/`、上传文件、工作区和报告的增长情况。由于上传和报告生成可能占用大量空间，应在磁盘耗尽前发出警报。

## 11. Python 客户端安装与身份验证

`scai-client` 版本 `0.2.0` 要求 Python `>=3.10` 和 `httpx>=0.26.0`。从仓库 checkout 安装：

```bash
python3 -m pip install ./clients/python
```

在用户界面中创建 API 密钥并初始化客户端：

```python
from scai_client import Client

client = Client(
    "http://localhost:3000",
    api_key="<API_KEY>",
    timeout=3600,
)
```

构造函数：`Client(base_url: str, *, api_key: str | None = None, jwt: str | None = None, timeout: int = 3600)`。至少需要一种凭据；两者均提供时使用 `jwt`。`base_url` 会移除尾部斜杠。`timeout` 配置底层同步 HTTP 客户端。

使用上下文管理器自动关闭连接：

```python
with Client("http://localhost:3000", api_key="<API_KEY>") as client:
    sessions = client.list_sessions()
```

常规生命周期公共方法为 `close() -> None`、`__enter__() -> Client` 和 `__exit__(*args) -> None`。凭据缺失或 HTTP 401 响应会引发 `AuthError`。

## 12. Python 客户端分析作业与产物

### 客户端作业方法

`submit(config_path: str | Path) -> Job` 读取 ConfigV1 JSON 文档、创建分析、上传每个存在的 `data.files[].local_path`（相对路径从配置目录解析），然后启动作业。无效配置会引发 `ScaiError`；身份验证失败会引发 `AuthError`。**缺失的被引用本地文件会被静默跳过，客户端仍会发送启动请求。**提交前请预检每个已配置路径；不要假设服务器会拒绝未携带预期文件而启动的作业。上传使用 10 MiB 分块，每个分块在首次尝试后最多重试三次，并由服务器端执行 SHA-256 验证。

`get_job(job_id: str) -> Job` 返回句柄，且不会预先检查其是否存在。

```python
from scai_client import Client

with Client("http://localhost:3000", api_key="<API_KEY>") as client:
    job = client.submit("analysis-config.json")
    final = job.wait(interval=60, timeout=24 * 3600)
    print(final["status"])
    print(job.artifacts())
    job.download_artifact("report.html", "downloads/report.html")
    job.download_export("downloads/job-export.zip")
```

### Job 构造函数与方法

`Job(client: httpx.Client, job_id: str)` 通常由 `Client` 创建；可通过 `job.job_id` 获取 `job_id`。

| 方法                                    | 行为与返回值                                                                                                           |
| --------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `status() -> dict`                    | 获取当前作业元数据。                                                                                                   |
| `wait(interval: int = 60, timeout: int  | None = None) -> dict`                                                                                                  |
| `artifacts() -> list`                 | 返回当前状态响应的`artifacts` 列表；若无则返回 `[]`。                                                              |
| `download_artifact(name: str, dest: str | Path) -> Path`                                                                                                         |
| `download_export(dest: str              | Path) -> Path`                                                                                                         |
| `delete() -> dict`                    | 仅当没有活动或正在初始化的工作流拥有该作业时才能删除。服务器可能返回 HTTP 409；请停止分析并等待初始化/清理结束后重试。 |

所有 Job HTTP 调用在收到 HTTP 401 时引发 `AuthError`，其他失败响应可能引发 `httpx.HTTPStatusError`。

客户端异常包括 `ScaiError`（基类）、`AuthError`、`FileIntegrityError`、`UploadError`、`JobFailed` 和 `ImportFailed`。文件访问也可能引发标准 `OSError`/`FileNotFoundError`；JSON 解析可能引发 `json.JSONDecodeError`；轮询可能引发 `TimeoutError`；网络故障可能表现为 `httpx` 异常。

## 13. Python 客户端可移植会话管理

`list_sessions() -> list[dict]` 返回经身份验证用户拥有的会话，包括标识符、名称、状态、轮次和时间戳。

常规导入用法：

```python
from scai_client import Client

def progress(info: dict) -> None:
    print(info.get("phase"), info.get("bytes_processed"), info.get("bytes_total"))

with Client("http://localhost:3000", api_key="<API_KEY>", timeout=86400) as client:
    session_id = client.import_session(
        "portable-session.tar.gz",
        wait=True,
        poll_interval=2.0,
        timeout=None,
        on_progress=progress,
    )
```

常规签名：`import_session(archive_path: str | Path, *, wait: bool = True, poll_interval: float = 2.0, timeout: int | None = None, on_progress: Callable[[dict], None] | None = None) -> str`。服务器接受 `.tar.gz`、`.tgz`、`.zip` 和 `.json` 格式。Tar 归档可以异步处理：设置 `wait=False` 时，返回字符串为导入作业 ID。进度回调异常会被忽略。提取进度将未压缩成员字节数与压缩归档字节数比较，因此该比率是启发式值，可能超过 100%。

**警告：**常规导入只在归档记录的会话 ID 于目标实例上未被占用时沿用该 ID。若已有会话占用它，归档会在新分配的 ID 下落地，并将其中所有嵌入路径重新绑定到新 ID，绝不覆盖既有会话的文件——导入始终是「创建会话」，而非「覆盖会话」。因此 ID 冲突的代价是换一个新 ID 以及失去精确续跑（见下文运行时段落），而不是数据丢失，无需为防冲突做完整备份或使用隔离实例。Tar 导入没有配置压缩大小、展开后大小或成员数量限制。请通过带外方式检查归档清单，并预留足够的数据/临时磁盘和 inode，以承受极端膨胀或归档炸弹。内联 ZIP 和 JSON 导入的请求体限制为 100 MiB，但该限制并不能保证归档内容安全。错误包括 `FileNotFoundError`、`AuthError`、`ScaiError`、`ImportFailed`、`TimeoutError` 以及 HTTP/网络错误。

`cancel_import(import_id: str) -> dict` 会在上传处理推进时以及归档成员之间请求协作式取消；取消并非即时事务回滚。状态查询与取消查询都按发起该作业的用户限定；未知 ID 与他人的 ID 一律返回 HTTP 404，因此该端点无法用于探测正在进行的导入，仅凭 ID 也无法操作他人的作业。该方法返回服务器响应，并会因身份验证失败、未知 ID 或其他 API 故障引发 `AuthError` 或 `ScaiError`。

常规导出用法：

```python
with Client("http://localhost:3000", api_key="<API_KEY>") as client:
    path = client.export_session(
        "<SESSION_ID>",
        "exports/portable-session.tar.gz",
        timeout=None,
        chunk_size=1024 * 1024,
    )
```

常规签名：`export_session(session_id: str, output_path: str | Path | None = None, *, timeout: float | None = None, chunk_size: int = 1048576) -> Path`。默认目标为 `<FIRST_8_ID_CHARS>_export.tar.gz`；传入目录时，该名称会放在目录内。数据先写入 `.partial` 文件，失败时删除，成功时以原子方式重命名。HTTP 401 引发 `AuthError`；所有权/未找到错误引发 `ScaiError`；其他 HTTP 故障引发 `httpx.HTTPStatusError`。

服务器会先完整构建临时压缩 tar 文件，然后才发送第一个响应字节。请确保 `upload_temp/` 有空间存放这一额外归档，并为大型会话预留足够的首字节等待时间；`timeout=None` 会为该请求禁用读/写/pool 超时，但保留客户端连接超时。

可移植导出包括选定的会话元数据、上传文件、工作区文件（包括图形/产物）、报告、可移植 notebook/状态 sidecar、外置内容、会话级 LLM 调用日志（`sessions/<SESSION_ID>/logs/`）和可选的可重现性 bundle。日志在任何情况下都会随归档传输：它是模型调用用量的唯一逐次记录，缺少它重新生成的报告会声称所有 agent 的调用次数均为零。每个导出的状态 JSON 都会剥离明文凭据与 endpoint；但设置 `include_runtime=true` 时，LangGraph checkpoint 中仍可能残留由源实例密钥封装的凭据——目标实例无法解密，导入时会将其丢弃。

导入后能否**继续分析**，取决于导出时的一个选择——若选择携带运行时，还取决于导入方是否信任该归档：

- **默认（`include_runtime=false`）**——不含内核 pkl、LangGraph checkpoint 和 h5ad 运行时 sidecar。数据、代码、图表、报告和对话历史都会保留且可查看，但会话无法继续：导入时其运行时会被标记为 `unavailable`，常规导出仍会携带的逐轮生信 checkpoint 会在导入时被删除，继续操作会提示需要从导入的数据重跑——会话随后正是这样做的：重新开始当前轮次，而不是恢复一段内核已经不存在的对话。适合把会话交给他人，或只关心结果的场景。
- **携带运行时（`include_runtime=true`）**——额外包含在线内核 pkl（`adata` 等内存变量）、LangGraph checkpoint，以及 Snapshot V2 恢复时可能代替 pkl 写出的 h5ad 运行时 sidecar，分析可以从中断处接着跑。导入时必须传 `X-Trust: true` 才会接受 pkl 与 h5ad sidecar，因为加载 pkl 会执行归档中的代码。不带该 header 时归档仍可导入，但不会落地任何在线内核，两类 checkpoint——LangGraph checkpoint 与逐轮生信 checkpoint——都会被丢弃，结果与上面的默认情形完全相同。归档体积可能大数 GB。

要获得精确续跑，需要三个条件同时成立：导出时 `include_runtime=true`、导入时 `X-Trust: true`，以及归档记录的会话 ID 在目标实例上仍未被占用。携带运行时的归档在**其他实例**上同样可以恢复，但最后一个条件才是悄悄决定「恢复得有多精确」的那一项。导入永不覆盖既有会话：该 ID 若已被占用，导入会另行分配一个新 ID，而以归档会话 ID 为 thread_id、无法改名落位的 LangGraph checkpoint 会随之被丢弃。此时内核 pkl 仍会恢复，会话保留内存变量，但「继续」只能从新一轮的起点重新进入工作流，而不是回到原先暂停的节点。因此把携带运行时的归档重新导入到导出它的那台实例上，只有先删除原会话才能获得精确续跑。两处与安装绑定的机制在导入时会被处理：pkl 的 HMAC sidecar 由目标实例用本地密钥重新生成（它从不随归档传输），源实例密钥封装的凭据会被丢弃而不是当作损坏——导入后重新配置 API key 即可。以上均不需要启用 Snapshot V2。

要能继续，会话必须是在暂停或等待输入时导出的。已完成的会话导出后，其 checkpoint 停在流程图末端，没有可继续的内容；此时应发起重新分析，开启新一轮。

它仍然是一种迁移格式，并非完整服务备份。

## 14. Session Manager CLI

使用仓库辅助程序前，请安装本地客户端包：

```bash
python3 -m pip install ./clients/python
python3 session_manager.py --url http://localhost:3000 --api-key '<API_KEY>' list
```

全局选项为 `--url URL`（默认 `http://localhost:3000`）和必需的 `--api-key API_KEY`。辅助程序创建 `Client(..., timeout=86400)`。

常规命令如下：

```bash
python3 session_manager.py --api-key '<API_KEY>' list
python3 session_manager.py --url http://localhost:3000 --api-key '<API_KEY>' \
  import '<ARCHIVE_PATH>'
python3 session_manager.py --url http://localhost:3000 --api-key '<API_KEY>' \
  export '<SESSION_ID>'
python3 session_manager.py --url http://localhost:3000 --api-key '<API_KEY>' \
  export '<SESSION_ID>' '<OUTPUT_PATH>'
```

`list` 显示 ID、名称、状态、当前轮次和创建时间。`import` 显示启发式提取进度（可能超过 100%），并在没有轮询期限的情况下等待；HTTP 客户端超时为 24 小时。它只在归档的会话 ID 未被占用时沿用该 ID，冲突时会另行分配新 ID，不会覆盖既有会话。`export` 写入压缩归档；省略输出时使用客户端默认名称；服务器需要临时磁盘在下载开始前完成构建。请引用 secrets 和路径，并避免将密钥或异步导入作业 ID 存入 shell 历史记录或日志。

## 15. 安全实践

- 通过受保护的环境注入设置强且稳定的 `SCAI_JWT_SECRET` 和 `SCAI_CONFIG_SECRET`。需要接受相同令牌的共享实例必须使用相同的 JWT 密钥。
- 切勿随意更改 `SCAI_CONFIG_SECRET`：先前存储的加密模型凭据可能无法读取。请将其与数据分开备份，并给予同等保护。
- 以最小权限的所有权和权限保护 `.jwt_secret`、`.config_secret`、`app.db`、归档、备份及所有数据目录。
- 以非特权账户运行应用进程。容器在准备卷之后切换到 UID `10001`。
- 将 API 密钥、JWT、模型提供商密钥和代理凭据视为 secrets。不要将其放入源码管理、命令记录、工单或日志；轮换并撤销已暴露的值。
- 根据用户需求限制对生产端口 `3000` 的网络访问；开发模式还需限制后端端口 `8000`。若无明确的网络控制方案，不要将直接启动绑定到 `0.0.0.0`。
- 向外部模型服务发送 prompts、图像或衍生研究内容前，应评估隐私与监管影响。确认提供商的数据保留和区域政策。
- 验证归档来源与内容，按照机构政策扫描传入文件，并仅导入到适当隔离的服务账户。
- 对静态和传输中的备份进行加密，在需要时保留离线/不可变副本，并定期测试恢复。
- 在 staging 和备份验证后，保持主机 OS、Docker、Python/Node 依赖项及 ScAI 镜像为最新版本。

## 16. 故障排除

| 症状                         | 检查与纠正操作                                                                                                                                                                       |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 容器持续不健康               | 运行`docker logs scai`；检查 `/data/logs/scai.log`；验证依赖项安装、磁盘空间、DNS/提供商访问，并在容器内使用 `curl` 访问后端健康端点。                                         |
| 前端可打开但 API 调用失败    | 生产模式下检查单一进程并运行 `curl http://127.0.0.1:3000/api/health`；开发模式下检查后端端口与 Vite 代理变量。同时查看浏览器/服务器日志。 |
| 直接启动提示实例正在运行     | 读取`.scai.pid` 并验证每个 PID。在专用主机上运行 `./stop.sh`；在共享主机上仅停止已验证的 PID。只有全部记录的服务均已停止时才能删除 PID 文件。                                    |
| 地址已被占用                 | 使用`lsof -i:<PORT>`，识别所有者，仅停止目标进程，或选择 `--port`/开发模式的 `--backport`。`stop.sh` 会清理已记录的自定义端口，但也可能终止占用这些端口的无关进程。 |
| 未找到`python3.12`         | 安装 Python 3.12 及其 venv 支持；`start.sh` 会先检查通用 `python3`，但首次创建环境时会明确调用 `python3.12`。                                                                  |
| Node/npm 或软件包安装失败    | 已打包的运行时不使用 Node。对源码构建或开发模式，请验证 Node.js 20.19+ 或 22.12+、npm、注册表访问、代理/index 设置、可用空间及依赖项目录所有权。重试前保留日志。 |
| 模型配置已保存但分析失败     | 保存不会验证凭据。使用验证操作，确认 base URL/代理/密钥、模型可用性、配额和出站连接。                                                                                                |
| 存储的模型密钥消失或无法解密 | 从受保护备份恢复原始`SCAI_CONFIG_SECRET` 或 `.config_secret`，然后重启。诊断期间不要覆盖数据库。                                                                                 |
| 重启后现有登录令牌停止工作   | 恢复稳定的`SCAI_JWT_SECRET` 或 `.jwt_secret`。用户可重新登录以获取新 JWT；API 密钥由数据库支持。                                                                                 |
| 上传/导入失败                | 检查支持的格式、源可读性、ID 冲突、数据/临时可用空间和 inode、客户端/服务器日志、HTTP 超时、所有权、成员数/膨胀情况及归档完整性。                                                    |
| 客户端收到 401               | 确认密钥前缀/值、所有权、撤销状态、账户状态、服务器 URL，以及 Authorization header 能否到达后端。                                                                                    |
| 客户端作业等待永不结束       | `Job.wait` 仅识别 `completed` 和 `failed`；检查 `job.status()` 中的 `stopped`、`paused` 和 `waiting_*`，查看服务器/提供商日志，设置有限超时，并处理 `TimeoutError`。 |
| 删除失败                     | 工作流处于活动或初始化状态时，`Job.delete` 返回 HTTP 409。请停止工作流并等待清理。用户删除和角色更改也会实施自身/最后管理员保护。                                                  |
| 磁盘用量增长                 | 检查上传文件、工作区、报告、日志、临时上传、遗留的`.partial` 文件和保留的会话。通过应用删除受支持的数据前请先备份。                                                                |

## 17. 参数与默认值参考

### 端口、路径与限制

| 项目               | 默认值/值            | 范围                             |
| ------------------ | -------------------- | -------------------------------- |
| 生产应用端口     | `3000`             | Docker 与直接启动的 UI/API/WebSocket |
| 开发后端端口     | `8000`             | 仅直接启动的开发模式          |
| Docker 映射        | `3000:3000`        | 推荐的主机映射                   |
| Docker 数据/依赖项 | `/data`, `/dep`  | 持久卷                           |
| 直接运行依赖项     | `<REPO>/../dep`    | `start.sh`                     |
| 直接运行数据       | `<REPO>/../data`   | 除非环境覆盖后端根目录           |
| PID 文件           | `<REPO>/.scai.pid` | 直接启动                         |
| 上传分块           | `10 MiB`           | 后端/客户端上传                  |
| 最大上传量         | `10 GiB`           | 后端                             |
| JWT 生命周期       | `72 hours`         | 登录令牌                         |
| 客户端 HTTP 超时   | `3600 seconds`     | `Client` 默认值                |
| 作业轮询间隔       | `60 seconds`       | `Job.wait` 默认值              |
| 导入轮询间隔       | `2.0 seconds`      | `Client.import_session` 默认值 |
| 导出流分块         | `1 MiB`            | `Client.export_session` 默认值 |

### 环境变量

| 变量                                    | 默认值                      | 效果                                                                                                       |
| --------------------------------------- | --------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `SCAI_DATA_DIR`                       | 解析后的同级`data` 目录   | 后端数据根目录；Docker 入口程序强制设为`/data`。覆盖后，直接运行的 `start.sh` 仍会创建/打印同级 data。 |
| `SCAI_JWT_SECRET`                     | 生成并存储为`.jwt_secret` | HS256 JWT 签名                                                                                             |
| `SCAI_CONFIG_SECRET`                  | 生成的本地加密材料          | 加密存储的用户/模型 secrets                                                                                |
| `SCAI_ENABLE_REPRODUCIBILITY`         | Disabled                    | 真值`1`、`true` 或 `yes` 启用可重现性 bundles                                                        |
| `SCAI_ENABLE_SNAPSHOT_V2`             | Disabled                    | 真值`1`、`true` 或 `yes` 为该实例启用 Snapshot V2                                                    |
| `SCAI_SNAPSHOT_SESSION_QUOTA_GB`      | `500`                     | 每个会话的快照存储上限，单位 GiB。**不建议设置为小于 100 GiB** —— 参见[快照容量](#快照容量)         |
| `SCAI_SNAPSHOT_FREE_SPACE_RESERVE_GB` | `50`                      | 一次捕获完成后数据卷上必须保留的剩余空间，单位 GiB                                                         |
| `SCAI_SNAPSHOT_MEMORY_SAFETY_RATIO`   | `0.8`                     | 一次捕获可规划使用的内核内存余量比例 —— 参见[快照的内存影响](#快照的内存影响)。超出 `(0, 1]` 的取值会被忽略并告警 |
| `SCAI_RUNTIME_MAX_RSS_GB`             | `auto`                    | 常规运行时持久化的逐会话内核 RSS 上限；数值表示管理员 GiB 上限                                              |
| `SCAI_RUNTIME_MAX_SINGLE_OBJECT_GB`   | `auto`                    | 最大存活对象上限；数值表示管理员 GiB 上限                                                                   |
| `SCAI_RUNTIME_MAX_ANNDATA_TOTAL_GB`   | `auto`                    | 存活 AnnData 估算总量上限；数值表示管理员 GiB 上限                                                          |
| `SCAI_RUNTIME_MAX_PKL_GB`             | `auto`                    | 常规压缩运行时文件上限；数值表示管理员 GiB 上限                                                             |
| `SCAI_RUNTIME_MAX_NAMESPACE_COUNT`    | `1500`                    | 常规持久化前允许测得的最大存活命名空间条目数                                                                |
| `SCAI_RUNTIME_MAX_SAVE_SECONDS`       | `300`                     | 常规运行时序列化的最长秒数                                                                                  |
| `SCAI_MODE`                           | `prod`                    | Docker 入口程序模式；生产镜像会拒绝 `dev`                                                        |
| `PIP_INDEX_URL`                       | pip environment/default     | Docker Python 软件包 index                                                                                 |
| `SCAI_AUTO_FETCH_MARKERS`             | `0`                       | 直接启动；`1` 尝试下载标记数据库                                                                         |
| `SCAI_HOST`                           | Set internally              | Vite 代理主机；脚本根据启动上下文设置                                                                      |
| `SCAI_BACKPORT`                       | Set internally              | Vite 代理后端端口；脚本根据启动上下文设置                                                                  |

### 快照容量

Snapshot V2 有两道上限，均在捕获写入之前检查一次、写入落盘之后再检查一次：

- **单会话存储配额** —— `SCAI_SNAPSHOT_SESSION_QUOTA_GB`，默认 500 GiB。以该会话
  对象库中的物理字节数计算。存储是内容寻址的，因此未发生变化的 AnnData 只保存一
  份、由引用它的所有快照共享；真正增长该数值的是新数据，而不是快照的数量。
- **数据卷剩余空间保留** —— `SCAI_SNAPSHOT_FREE_SPACE_RESERVE_GB`，默认 50 GiB。
  一次捕获完成后仍必须保持空闲的空间，以免单个会话占满所有会话都要写入元数据的
  那个卷。

两者都在后端启动时读取一次；请在启动前设置，修改后需重启生效。

**请勿将单会话配额设置为小于 100 GiB。** 一次大型 AnnData 的运行时捕获本身就可达
数十 GB，而存储会保留该会话快照谱系中每一个不同的对象，而非仅保留最新的一份。小
于这个量级的配额不会表现为性能逐步下降：越过上限的那次捕获会 fail-closed，并将分
析停在快照失败门上，需用户处理后才能继续。小于 100 GiB 的取值仍会生效，但启动时会
记录一条告警。

配额是按会话计的，因此最坏情况下的总量为配额乘以持有快照的会话数；剩余空间保留是
唯一的卷级约束。请按预期的并发分析数量来规划卷容量，并注意垃圾回收是在删除快照时
触发的，而非定时执行 —— 被取代对象所占空间在删除时回收。

当某个会话占用达到其配额的 90% 时，快照面板会提示用户清理，因此通常会在接近上限前
得到预警，而不是毫无察觉地撞上。

### Shell 选项

| 程序                          | 选项                                | 默认值/效果                                               |
| ----------------------------- | ----------------------------------- | --------------------------------------------------------- |
| `start.sh`                  | `--dev`                           | 开发模式                                                  |
| `start.sh`                  | `--prod`                          | 生产模式；默认值                                          |
| `start.sh`                  | `--host HOST`                     | `127.0.0.1`                                             |
| `start.sh`                  | `--port PORT`                     | 生产应用/开发前端`3000`                                  |
| `start.sh`                  | `--backport PORT`                 | 开发后端`8000`；生产模式忽略                             |
| `start.sh`                  | `-h`, `--help`                  | 帮助                                                      |
| `stop.sh`                   | none                                | 清理记录的 PID/端口，并执行宽泛的主机进程名称后备匹配     |
| `build.sh`                  | `--version TAG`                   | 最新 Git 标签，否则为`latest`                           |
| `build.sh`                  | `--push`                          | 推送所选标签和`latest`；默认禁用                        |
| `build.sh`                  | `-h`, `--help`                  | 帮助                                                      |
| `session_manager.py`        | `--url URL`                       | `http://localhost:3000`                                 |
| `session_manager.py`        | `--api-key KEY`                   | 必需                                                      |
| `session_manager.py list`   | none                                | 列出拥有的会话                                            |
| `session_manager.py import` | `archive`                         | 必需的归档路径                                            |
| `session_manager.py export` | `session_id`, optional `output` | 必需 ID 和可选目标                                        |

### Python 常规公共 API

| 类型       | 签名                                                                                              |
| ---------- | ------------------------------------------------------------------------------------------------- |
| `Client` | `Client(base_url, *, api_key=None, jwt=None, timeout=3600)`                                     |
| `Client` | `close()`                                                                                       |
| `Client` | `submit(config_path)`                                                                           |
| `Client` | `get_job(job_id)`                                                                               |
| `Client` | `list_sessions()`                                                                               |
| `Client` | `import_session(archive_path, *, wait=True, poll_interval=2.0, timeout=None, on_progress=None)` |
| `Client` | `export_session(session_id, output_path=None, *, timeout=None, chunk_size=1048576)`             |
| `Client` | `cancel_import(import_id)`                                                                      |
| `Job`    | `Job(client, job_id)`                                                                           |
| `Job`    | `status()`                                                                                      |
| `Job`    | `wait(interval=60, timeout=None)`                                                               |
| `Job`    | `artifacts()`                                                                                   |
| `Job`    | `download_artifact(name, dest)`                                                                 |
| `Job`    | `download_export(dest)`                                                                         |
| `Job`    | `delete()`                                                                                      |
