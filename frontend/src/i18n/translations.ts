// Internationalization translations

export type Language = 'zh' | 'en'

export const translations = {
  zh: {
    // App
    appName: 'ScAI',
    appSubtitle: '单细胞测序AI分析',
    appFullName: 'ScAI - 单细胞测序AI分析',
    loadingAnalysis: '正在加载分析…',

    // Tabs
    tabAnalysis: '分析',
    tabData: '数据',
    tabPrompts: 'Agents',
    tabResults: '结果',
    tabReport: '报告',
    tabChat: '对话',
    
    // Sidebar
    newAnalysis: '新建分析',
    defaultProjectName: '新分析项目',
    defaultProjectDescription: '请描述您的研究目标和实验设计...',
    exportState: '导出状态',
    importState: '导入状态',
    exportSession: '导出会话',
    importSession: '导入会话',
    importing: '导入中...',
    exporting: '导出中...',
    exportPackaging: '正在打包...',
    exportPackagingDetail: '正在压缩文件，请稍候...',
    exportDownloadStarted: '已经开始下载',
    exportDownloadStartedDetail: '文件下载已开始，请检查浏览器下载管理器',
    importSuccess: '导入成功',
    importFailed: '导入失败',
    exportFailed: '导出失败',
    confirm: '确定',
    apiSettings: 'API 设置',
    
    // Config Panel
    apiConfig: 'API 配置',
    apiConfigDesc: '配置各平台的 API 密钥',
    openrouterKey: 'OpenRouter API Key',
    openrouterDesc: '用于调用 Gemini 和 Claude 模型',
    googleKey: 'Google AI API Key',
    googleDesc: '用于 Gemini Deep Research',
    getKey: '获取 Key',
    required: '必填',
    cancel: '取消',
    saveConfig: '保存配置',
    saving: '保存中...',
    saved: '已保存',
    readMode: '阅读模式',
    readModeDesc: '浏览已有分析结果（无需 API Key）',
    validatingApiKey: '正在验证 API Key...',
    apiKeyInvalid: 'API Key 验证失败',
    apiKeyInvalidDesc: '无法通过 API Key 连接到 LLM 服务器。可能是 Key 无效或服务器不可达。',
    apiKeyValidationUncertain: '无法确认 API Key 是否有效',
    apiKeyValidationUncertainDesc: '无法连接到 LLM 服务器进行验证，但这可能是网络问题。',
    reconfigureApiKey: '重新设置 API Key',
    ignoreError: '忽略错误',
    readModeRestriction: '阅读模式下无法启动新分析，请先配置 API Key',
    
    // New Session Modal
    createAnalysis: '新建分析',
    createAnalysisDesc: '创建新的分析会话',
    sessionName: '分析名称（可选）',
    sessionNamePlaceholder: '例如：单细胞测序分析-实验1',
    researchGoal: '研究目标',
    researchGoalPlaceholder: '例如：分析特定基因敲除小鼠的组织表型，关注细胞类型变化和相关信号通路...',
    create: '创建',
    
    // Analysis Panel
    analysisControl: '分析控制',
    startAnalysis: '开始分析',
    pause: '暂停',
    resume: '继续',
    connected: '已连接',
    analyzing: '分析进行中',
    analyzingDesc: '各Agent正在协作分析...',
    analysisComplete: '分析完成',
    analysisFailed: '分析失败',
    checkConfigAndRetry: '请检查配置和数据后重试',
    agentStatus: 'Agent 状态',
    analysisRounds: '分析轮次',
    findings: '个发现',
    newSession: '新建',
    
    // Agent names
    piAgent: 'PI Agent',
    piAgentDesc: '项目负责人，协调各方',
    pathologyAgent: '病理 Agent',
    pathologyAgentDesc: '分析组织学切片',
    bioinformaticsAgent: '生信 Agent',
    bioinformaticsAgentDesc: '单细胞测序分析',
    biochemistryAgent: '生化 Agent',
    biochemistryAgentDesc: '生化指标解读',
    literatureAgent: '文献 Agent',
    literatureAgentDesc: '文献调研支持',
    literatureResearchMode: '搜索模式',
    literatureGeminiDeepResearch: 'Gemini Deep Research',
    literatureGeminiDesc: '使用 Google Gemini Deep Research API 进行综合文献检索',
    literatureGeminiNote: '需要在 API 设置中配置 Google AI API Key',
    literatureDeepResearch: 'Deep Research Models',
    literatureDeepResearchDesc: '使用具有深度研究能力的 LLM 模型进行文献检索，如 o4-mini-deep-research、sonar-deep-research',
    literatureDeepResearchModel: 'Deep Research 模型',
    literatureDeepResearchModelPlaceholder: '如 o4-mini-deep-research',
    literatureDeepResearchNote: '响应时间较长（5-30分钟）。推荐模型: o4-mini-deep-research、sonar-deep-research',
    literatureDeepResearchServer: 'LLM 服务器',
    literatureDeepResearchServerHint: '留空则使用 PubMed LLM 的服务器和 API Key',
    literatureDeepResearchApiKey: 'API Key',
    literaturePubmedLlm: '仅 PubMed + LLM',
    literaturePubmedDesc: '仅使用 PubMed 数据库搜索文献 + LLM 综合分析',
    literaturePubmedEmailLabel: 'PubMed 邮箱',
    literaturePubmedEmailNote: 'NCBI E-utilities 要求提供联系邮箱',
    literatureGoogleApiKey: 'Google AI API Key',
    literatureMaxSearchRounds: '最大搜索轮数',
    literatureMaxSearchRoundsNote: 'Agent 可根据搜索结果动态调整查询，进行多轮迭代搜索',
    literaturePubmedSettings: 'PubMed 搜索设置',
    literaturePubmedSettingsNote: '所有模式在搜索失败时都会回退到 PubMed 检索',
    painterAgent: '绘图 Agent',
    painterAgentDesc: '图形摘要绘制',
    supervisorAgent: '监督 Agent',
    supervisorAgentDesc: '实时分析监控与质量控制',
    techniqueAgent: '技术 Agent',
    techniqueAgentDesc: 'API文档查询与错误摘要',

    // Publication figures and reports
    figureGallery: '图片库',
    figureApproved: '已通过',
    figurePending: '待审核',
    figureRevisionRequested: '需修改',
    graphicAbstract: '图形摘要',
    graphicAbstractTextMode: '图形摘要 (文字模式)',
    graphicAbstractGenerated: '图形摘要生成',
    integratedReport: '整合报告',
    integratedReportGenerated: '整合报告生成',
    downloadPDF: '下载 PDF',
    downloadPNG: '下载 PNG',
    downloadText: '下载文本',
    downloadHD: '下载高清图',
    downloadAllFigures: '下载所有图片',
    downloadAllTables: '下载所有表格',
    generatingGraphicAbstract: '正在生成图形摘要...',
    graphicAbstractPrompt: '生成提示: ',
    generatedAt: '生成于 ',
    clickToEnlarge: '点击放大',
    noFiguresYet: '暂无图片',
    noTablesYet: '暂无表格',
    piReviewingFigure: 'PI 正在审核图片...',
    figureApprovedSuccess: '图片已通过审核',
    figureNeedsRevision: '图片需要修改',
    qualityScore: '质量评分',
    
    // Data Panel
    pathologySlides: '病理切片',
    pathologySlidesDesc: '上传 H&E 或特殊染色切片',
    dragOrClick: '拖拽文件到这里，或点击选择',
    supportedFormats: '支持 PNG, JPEG, TIFF 格式',
    sequencingData: '测序数据',
    sequencingDataDesc: '上传单细胞或 Bulk 测序数据',
    uploadSequencing: '点击或拖拽上传测序数据',
    supportedH5ad: '支持 .h5ad, .h5 (scRNA-seq), .csv/.tsv (bulk RNA-seq) 或 .loom (RNA velocity)',
    dataUploaded: '数据已上传',
    biochemistryIndicators: '生化指标',
    biochemistryIndicatorsDesc: '上传数据文件或手动输入生化指标',
    addIndicator: '添加指标',
    indicatorName: '指标名称 (如 ALT)',
    value: '数值',
    unit: '单位 (如 U/L)',
    add: '添加',
    noBiochemistryData: '暂无生化数据',
    indicator: '指标',
    uploadData: '上传分析数据',
    uploadDataDesc: '请先创建一个分析会话，然后上传病理切片、单细胞测序数据和生化指标。',

    // Treatment groups
    treatmentGroups: '处理组',
    treatmentGroupsDesc: '定义实验的处理组和样本数',
    addGroup: '添加处理组',
    groupName: '组名',
    groupNamePlaceholder: '如：对照组、模型组',
    groupDescription: '描述',
    groupDescPlaceholder: '详细描述（可选）',
    sampleCount: '样本数',
    noGroupDistinction: '不区分处理组',
    noSampleDistinction: '不区分样本',
    deleteGroupConfirm: '删除此处理组将同时移除其关联的所有数据，确定？',
    noGroups: '请先添加处理组',

    // Data section
    selectGroup: '选择处理组',
    selectSample: '选择样本',
    selectDataType: '选择数据类型',
    sample: '样本',
    filename: '文件名',
    group: '组别',
    dataType: '类型',
    actions: '操作',
    noFiles: '暂无文件',

    // Data overview
    dataOverview: '数据总览',
    dataOverviewDesc: '各处理组的数据上传情况',
    filesCount: '{n} 个文件',
    indicatorsCount: '{n} 项指标',

    // Stain types
    heStain: 'H&E 染色',
    siriusRed: 'Sirius Red (纤维化)',
    perlsBlue: 'Perls Blue (铁沉积)',
    oilRedO: 'Oil Red O (脂肪)',
    pas: 'PAS (糖原)',
    other: '其他',
    customType: '自定义类型',
    customTypePlaceholder: '例如 Sirius Red',
    customTypeRequired: '请输入自定义类型名称',

    // Sequencing types
    scRNASeq: 'scRNA-seq (h5ad/h5)',
    loom: 'Loom (RNA velocity)',

    // Biochemistry types
    bioImage: '图片',
    bioFile: '数据文件 (CSV/Excel)',
    
    // Prompts Panel
    selectAgent: '选择 Agent',
    agentLlmServer: 'LLM 服务器',
    agentApiKey: 'API Key',
    agentServerHint: '当前使用默认 LLM 服务器设置，如需为此 Agent 使用不同平台请修改',
    agentModelHint: '注意：不同平台的模型名格式可能不同（如 OpenRouter 使用 provider/model 格式）',
    agentDisableHint: '留空则不启用此 Agent',
    agentRequiredHint: '此 Agent 为必需，不可禁用',

    // LLM failure dialog
    llmCallFailed: 'LLM 调用失败',
    llmCallFailedBadge: 'LLM调用失败',
    llmAgent: '智能体',
    llmModel: '模型',
    llmServer: 'LLM 服务器',
    llmErrorDetails: '错误详情',
    llmRetry: '重试',
    llmRetryHint: '最多再试 3 次',
    llmModifyAgent: '修改 Agent 配置',
    llmTerminate: '终止分析',
    llmSkip: '跳过',
    llmAutoSkip: '{seconds} 秒后自动跳过',
    llmClose: '关闭',
    agentDisabledTitle: 'Agent 已禁用',
    agentDisabledBody: '当前流程需要 {agent}，但其 model 名为空。请选择：',
    agentDisabledModifyAgent: '修改 Agent（推荐）',
    llmReopenHint: '关闭弹窗后，可通过顶部 "LLM调用失败" 标签重新打开',
    configModified: '配置已修改',
    configModifiedDesc: 'Agent 配置已更新。请选择下一步操作：',
    resumeAnalysis: '恢复分析',
    returnToSettings: '返回修改',
    backToTimeoutDialog: '返回超时对话框',

    visionSupport: 'Vision 支持',
    visionSupportEnabled: '模型支持图片输入',
    visionSupportDisabled: '模型不支持图片输入',
    visionHint_pi: '启用后 PI 可直接审核图片质量；关闭后将由其他有 vision 能力的 Agent 代替审核',
    visionHint_bioinformatics: '启用后生信 Agent 可在修订图片时看到当前图片效果',
    visionHint_biochemistry: '若上传了生化结果图片，需要启用以识别图片内容',
    visionHint_painter: '启用后 Painter 修订模式图时可看到当前图片并进行针对性修改',
    visionHint_supervisor: '启用后 Supervisor 可实时检查生成图片的质量',
    visionHint_technique: '技术 Agent 通常不需要视觉能力',
    reset: '重置',
    save: '保存',
    promptTips: '提示词编写建议',
    promptTip1: '明确 Agent 的角色定位和专业背景',
    promptTip2: '列出需要关注的具体任务和指标',
    promptTip3: '说明与其他 Agent 的协作方式',
    promptTip4: '指定输出格式和语言偏好',
    
    // Results Panel
    analysisResults: '分析结果',
    completedRounds: '共完成 {n} 轮分析',
    downloadReport: '下载报告',
    finalReport: '简要分析结果',
    roundDetails: '各轮次详情',
    waitingResults: '等待分析结果',
    waitingResultsDesc: '分析完成后，各轮次的结果和最终报告将显示在这里。',
    pathologyFindings: '病理发现',
    bioinformaticsFindings: '生信发现',
    biochemistryFindings: '生化发现',
    literatureFindings: '文献发现',
    crossModalCandidates: '跨模态机制候选',
    crossModalConfidence: '证据强度',
    crossModalRelation: '与当前方向关系',
    
    // Message Stream
    realTimeMessages: '实时消息',
    agentInteractions: 'Agent 交互记录',
    waitingStart: '等待分析开始...',
    code: '代码',
    output: '输出',
    chart: '图表',
    
    // Welcome screen
    startNewAnalysis: '开始新的分析',
    welcomeDesc: '创建一个新的分析会话，上传数据后开始多智能体协作分析。支持病理切片、测序数据、生化指标的综合分析。',
    
    // Errors
    pleaseEnterOpenrouterKey: '请输入 OpenRouter API Key',
    saveFailed: '保存失败',
    
    // Default Prompts
    promptPi: `You are a senior biomedical research PI (Principal Investigator), leading a multidisciplinary team for comprehensive analysis.

Your responsibilities:
1. Develop overall research strategy and analysis direction
2. Coordinate work across pathology, bioinformatics, biochemistry, and literature teams
3. Synthesize results from all parties and make scientific judgments
4. Arbitrate conflicts and provide reasonable explanations
5. Identify novel scientific discoveries and potential mechanisms

Core issues to focus on:
- Overall phenotypic characteristics of the samples
- Degree and type of tissue damage
- Changes in cell composition
- Potential molecular mechanisms
- Consistency with known literature and new findings

Please analyze based on the specific research goals provided by the user. Respond in English and maintain scientific rigor.`,

    promptPathology: `You are a professional pathology expert, skilled in analyzing various histological slides.

Your tasks:
1. Analyze stained slides to identify tissue structural abnormalities
2. Assess lesion severity (mild/moderate/severe)
3. Identify concrete, stain-interpretable features that are visible in the supplied images
   and relevant to the stated study question; do not apply a fixed feature checklist
4. Recommend special staining when necessary

Please describe the pathological features you observe in detail and provide professional judgment. Respond in English.`,

    promptBioinformatics: `You are a professional bioinformatics analyst, specializing in high-throughput sequencing data analysis (including single-cell RNA-seq, bulk RNA-seq, etc.).

Your workflow:
1. Write and execute Python code step by step
2. Review results after each step before deciding the next
3. Request literature support when encountering important biological questions

Analysis pipeline (adjust based on data type):
1. Quality Control (QC)
2. Normalization and dimensionality reduction
3. Clustering analysis
4. Cell/sample annotation
5. Differential expression analysis
6. Pathway enrichment analysis
7. Other specialized analyses

Code requirements:
- Use appropriate analysis libraries (scanpy, pydeseq2, etc.)
- Set random_state=42 for reproducibility
- Generate visualizations at each step
- Save key intermediate results

Please respond with analysis conclusions in English, with code comments in English.`,

    promptBiochemistry: `You are a laboratory biochemistry expert, responsible for interpreting and integrating experimental results.

Base the analysis on the biochemical indicators actually supplied by the user. Organize them
by their measured role only after inspecting the data; do not assume a fixed panel.

Your tasks:
1. Integrate manually entered experimental data
2. Assess the degree of abnormality for each indicator
3. Hypothesize possible pathophysiological mechanisms
4. Recommend supplementary experiments

Please respond in English with professional analytical opinions.`,

    promptLiterature: `Please help me conduct a literature review on the topic: {topic}

Please search for relevant biomedical literature, focusing on:
1. Related molecular mechanisms
2. Known gene/pathway associations
3. Clinical/experimental relevance
4. Latest research developments

Please provide a detailed literature review report with specific citation information.`,

    // Round Feedback Dialog
    roundComplete: '第 {round} 轮分析完成',
    roundSummaryTitle: '本轮总结',
    keyFindings: '主要发现',
    additionalDataSuggested: '建议补充的数据',
    noFindings: '本轮无发现',
    autoProceeding: '即将自动继续...',
    timerPaused: '计时已暂停',
    skipRound: '继续',
    pauseTimer: '暂停',
    resumeTimer: '继续',
    sendInput: '发送',
    sendingInput: '发送中...',
    lastRoundNotice: '这是最后一轮分析。提供输入将触发额外的分析轮次。',
    selectTargetAgent: '目标 Agent',
    uploadFilesDialog: '上传文件',
    roundInputPlaceholder: '输入补充说明或对下一步分析的指导意见...',
    fileUploadedTo: '发送给',
    removeFile: '移除',

    // Chat Panel
    chatPlaceholder: '输入消息...',
    chatSend: '发送',
    chatEmpty: '开始与 PI Agent 对话',
    chatEmptyDesc: 'PI Agent 可以根据分析结果回答问题，也可以分析您上传的数据',
    chatNoSession: '请先创建或选择一个分析会话',
    chatError: '发送失败，请重试',
    chatStop: '停止',
    chatCancelled: '已取消',
    chatAttach: '附件',
    chatTargetAgent: '目标 Agent',
    chatTargetPi: 'PI（自动分发）',
    chatTargetPathology: '病理',
    chatTargetBioinformatics: '生信',
    chatTargetBiochemistry: '生化',

    // Supplementary Analysis
    requestMoreAnalysis: '请求补充分析',
    supplementaryAnalysisPrompt: '请描述您希望进行的补充分析',
    supplementaryAnalysisRunning: '正在执行补充分析...',
    supplementaryAnalysisComplete: '补充分析完成',
    supplementarySubmit: '提交',
    supplementaryPlaceholderAnnotation: '例如：请进一步分析cluster 3和cluster 5的差异基因，并搜索相关文献',
    supplementaryPlaceholderDirection: '例如：请进一步搜索关于XXX通路在这个组织中的文献，以帮助判断研究方向',
    supplementaryDesc: '请描述您希望进行的补充分析，PI Agent将组织各Agent执行分析后更新报告。',

    // Detailed Report
    viewDetailedReport: '查看详细报告',

    // Snapshots
    snapshots: '快照 (实验性)',
    createSnapshot: '创建快照',
    restoreSnapshot: '恢复快照',
    deleteSnapshot: '删除快照',
    confirmDelete: '确认删除',
    discardCurrent: '丢弃当前状态',
    confirmDiscard: '确认丢弃',
    restoreProgress: '恢复进度',
    snapshotCreated: '快照已创建',
    snapshotRestored: '已恢复到快照',
    confirmRestore: '确认恢复到此快照？当前分析进度将被替换。',
    manualSnapshot: '手动快照',
    noSnapshots: '暂无快照',
    snapshotName: '快照名称（可选）',
    phasePreAnnotation: '注释前',
    phasePostAnnotation: '注释后',
    phasePreDirection: '选方向前',
    phasePostDirection: '选方向后',
    phaseManual: '手动',
    analysisSnapshots: '分析快照 (实验性)',
    creating: '创建中...',

    // Plan Review Dialog
    planReviewTitle: '分析计划审核',
    planReviewDownloadLog: '下载对话记录',
    planReviewOverview: '概述',
    planReviewSteps: '分析步骤 ({n} 步)',
    planReviewMethod: '方法：',
    planReviewInput: '输入：',
    planReviewExpectedOutput: '预期输出：',
    planReviewRationale: '理由：',
    planReviewDependsOn: '依赖步骤：',
    planReviewKeyDecisions: '关键决策点',
    planReviewEstimatedTime: '预计运行时间:',
    planReviewPlaceholder: '输入对计划的修改建议...',
    planReviewCancel: '取消',
    planReviewSubmitting: '提交中...',
    planReviewSubmitFeedback: '提交修改意见',
    planReviewModify: '修改',
    planReviewSkip: '跳过（让 PI 自行决定）',
    planReviewApprove: '批准方案',
    planReviewStageExplore: '探索阶段',
    planReviewStageInnovate: '创新阶段',
    planReviewStage1: '阶段 1: QC+聚类+注释',
    planReviewStage2: '阶段 2: 下游分析',
    planReviewStageFull: '全量分析',

    integrationGateTitle: '整合分析 (第 {cycle} 次) — 选择新的研究方向',
    integrationGateCountdown: '倒计时:',
    integrationGateContinue: '继续分析所选方向',
    integrationGateLetPiDecide: '让 PI 决定',
    integrationGateFinalize: '进入报告撰写',
    integrationGateSelectRequired: '请至少选择一个研究方向。',
    integrationGateActionFailed: '操作失败：{message}',

    // Species input dialog
    speciesInputTitle: '物种信息',
    speciesGeneValidationTitle: '基因名验证',
    speciesInputPlaceholder: '请输入物种名称，如 Homo sapiens、Mus musculus...',
    speciesGeneValidationPlaceholder: '请输入正确的物种名称，或输入"保持当前设置"...',
    waitingSpeciesInput: '等待物种信息输入...',

    // Exploration failed dialog (likely a data problem)
    explorationFailedTitle: '探索阶段未成功',
    explorationFailedMessage: '探索阶段（Stage 1：质控/聚类/注释）未能成功完成。这通常是因为上传的数据存在问题（格式、内容或缺失字段）。请检查或重新上传数据后继续重试 Stage 1，或停止分析。',
    explorationFailedReasonLabel: '原因',
    explorationFailedReason_did_not_converge: '分析未能收敛（已达最大步数或未完成）',
    explorationFailedReason_analysis_error: '分析过程中发生错误',
    explorationFailedGoToData: '前往数据面板',
    explorationFailedContinue: '继续（重试 Stage 1）',
    explorationFailedStop: '停止分析',

    // Marker DB missing dialog
    markerDbMissingTitle: 'Marker 数据库下载失败',
    markerDbMissingDesc: 'Marker 数据库 TSV 自动下载失败。可重试下载以启用数据库先验，或跳过（仅用文献注释），或终止分析。',
    markerDbFetch: '重试下载',
  },

  en: {
    // App
    appName: 'ScAI',
    appSubtitle: 'Single Cell Sequencing Analysis Driven by AI',
    appFullName: 'ScAI - Single Cell Sequencing Analysis Driven by AI',
    loadingAnalysis: 'Loading analysis…',

    // Tabs
    tabAnalysis: 'Analysis',
    tabData: 'Data',
    tabPrompts: 'Agents',
    tabResults: 'Results',
    tabReport: 'Report',
    tabChat: 'Chat',

    // Sidebar
    newAnalysis: 'New Analysis',
    defaultProjectName: 'New Analysis Project',
    defaultProjectDescription: 'Describe your research goals and experimental design...',
    exportState: 'Export State',
    importState: 'Import State',
    exportSession: 'Export Session',
    importSession: 'Import Session',
    importing: 'Importing...',
    exporting: 'Exporting...',
    exportPackaging: 'Packaging...',
    exportPackagingDetail: 'Compressing files, please wait...',
    exportDownloadStarted: 'Download Started',
    exportDownloadStartedDetail: 'File download has started, please check your browser download manager',
    importSuccess: 'Import successful',
    importFailed: 'Import failed',
    exportFailed: 'Export failed',
    confirm: 'Confirm',
    apiSettings: 'API Settings',
    
    // Config Panel
    apiConfig: 'API Configuration',
    apiConfigDesc: 'Configure API keys for each platform',
    openrouterKey: 'OpenRouter API Key',
    openrouterDesc: 'For calling Gemini and Claude models',
    googleKey: 'Google AI API Key',
    googleDesc: 'For Gemini Deep Research',
    getKey: 'Get Key',
    required: 'Required',
    cancel: 'Cancel',
    saveConfig: 'Save Config',
    saving: 'Saving...',
    saved: 'Saved',
    readMode: 'Read Mode',
    readModeDesc: 'Browse existing analyses (no API Key needed)',
    validatingApiKey: 'Validating API Key...',
    apiKeyInvalid: 'API Key Validation Failed',
    apiKeyInvalidDesc: 'Could not connect to the LLM server with this API Key. The key may be invalid or the server unreachable.',
    apiKeyValidationUncertain: 'Unable to Verify API Key',
    apiKeyValidationUncertainDesc: 'Could not connect to the LLM server for validation, but this may be a network issue.',
    reconfigureApiKey: 'Reconfigure API Key',
    ignoreError: 'Ignore Error',
    readModeRestriction: 'Cannot start new analysis in Read Mode. Please configure API Key first.',
    
    // New Session Modal
    createAnalysis: 'New Analysis',
    createAnalysisDesc: 'Create a new analysis session',
    sessionName: 'Session Name (Optional)',
    sessionNamePlaceholder: 'e.g., Single-cell Analysis - Exp1',
    researchGoal: 'Research Goal',
    researchGoalPlaceholder: 'e.g., Analyze the tissue phenotype of gene knockout mice, focusing on cell type changes and related signaling pathways...',
    create: 'Create',
    
    // Analysis Panel
    analysisControl: 'Analysis Control',
    startAnalysis: 'Start Analysis',
    pause: 'Pause',
    resume: 'Resume',
    connected: 'Connected',
    analyzing: 'Analysis in Progress',
    analyzingDesc: 'Agents are collaborating on analysis...',
    analysisComplete: 'Analysis Complete',
    analysisFailed: 'Analysis Failed',
    checkConfigAndRetry: 'Please check configuration and data, then retry',
    agentStatus: 'Agent Status',
    analysisRounds: 'Analysis Rounds',
    findings: 'findings',
    newSession: 'New',
    
    // Agent names
    piAgent: 'PI Agent',
    piAgentDesc: 'Principal Investigator, coordinates all',
    pathologyAgent: 'Pathology Agent',
    pathologyAgentDesc: 'Analyzes histological slides',
    bioinformaticsAgent: 'Bioinformatics Agent',
    bioinformaticsAgentDesc: 'Single-cell sequencing analysis',
    biochemistryAgent: 'Biochemistry Agent',
    biochemistryAgentDesc: 'Interprets biochemical data',
    literatureAgent: 'Literature Agent',
    literatureAgentDesc: 'Literature research support',
    literatureResearchMode: 'Research Mode',
    literatureGeminiDeepResearch: 'Gemini Deep Research',
    literatureGeminiDesc: 'Comprehensive literature search via Google Gemini Deep Research API',
    literatureGeminiNote: 'Requires Google AI API Key (set in API Settings)',
    literatureDeepResearch: 'Deep Research Models',
    literatureDeepResearchDesc: 'Literature search via LLM models with deep research ability such as o4-mini-deep-research and sonar-deep-research',
    literatureDeepResearchModel: 'Deep Research Model',
    literatureDeepResearchModelPlaceholder: 'e.g., o4-mini-deep-research',
    literatureDeepResearchNote: 'Response time: 5-30 minutes. Recommended: o4-mini-deep-research, sonar-deep-research',
    literatureDeepResearchServer: 'LLM Server',
    literatureDeepResearchServerHint: 'Leave empty to use PubMed LLM server and API key',
    literatureDeepResearchApiKey: 'API Key',
    literaturePubmedLlm: 'PubMed + LLM Only',
    literaturePubmedDesc: 'Search PubMed database only + LLM synthesis and analysis',
    literaturePubmedEmailLabel: 'PubMed Email',
    literaturePubmedEmailNote: 'NCBI E-utilities requires a contact email',
    literatureGoogleApiKey: 'Google AI API Key',
    literatureMaxSearchRounds: 'Max Search Rounds',
    literatureMaxSearchRoundsNote: 'Agent dynamically adjusts queries and performs iterative multi-round search',
    literaturePubmedSettings: 'PubMed Search Settings',
    literaturePubmedSettingsNote: 'All modes fall back to PubMed search on failure',
    painterAgent: 'Painter Agent',
    painterAgentDesc: 'Graphic abstract generation',
    supervisorAgent: 'Supervisor Agent',
    supervisorAgentDesc: 'Real-time analysis monitoring and quality control',
    techniqueAgent: 'Technique Agent',
    techniqueAgentDesc: 'API documentation lookup & error summarization',

    // Publication figures and reports
    figureGallery: 'Figure Gallery',
    figureApproved: 'Approved',
    figurePending: 'Pending Review',
    figureRevisionRequested: 'Revision Requested',
    graphicAbstract: 'Graphic Abstract',
    graphicAbstractTextMode: 'Graphic Abstract (Text Mode)',
    graphicAbstractGenerated: 'Graphic Abstract Generated',
    integratedReport: 'Integrated Report',
    integratedReportGenerated: 'Integrated Report Generated',
    downloadPDF: 'Download PDF',
    downloadPNG: 'Download PNG',
    downloadText: 'Download Text',
    downloadHD: 'Download HD',
    downloadAllFigures: 'Download All Figures',
    downloadAllTables: 'Download All Tables',
    generatingGraphicAbstract: 'Generating graphic abstract...',
    graphicAbstractPrompt: 'Prompt: ',
    generatedAt: 'Generated at ',
    clickToEnlarge: 'Click to enlarge',
    noFiguresYet: 'No figures yet',
    noTablesYet: 'No tables yet',
    piReviewingFigure: 'PI reviewing figure...',
    figureApprovedSuccess: 'Figure approved',
    figureNeedsRevision: 'Figure needs revision',
    qualityScore: 'Quality Score',
    
    // Data Panel
    pathologySlides: 'Pathology Slides',
    pathologySlidesDesc: 'Upload H&E or special stain slides',
    dragOrClick: 'Drag files here, or click to select',
    supportedFormats: 'Supports PNG, JPEG, TIFF formats',
    sequencingData: 'Sequencing Data',
    sequencingDataDesc: 'Upload single-cell or bulk sequencing data',
    uploadSequencing: 'Click or drag to upload sequencing data',
    supportedH5ad: 'Supports .h5ad, .h5 (scRNA-seq), .csv/.tsv (bulk RNA-seq) or .loom (RNA velocity)',
    dataUploaded: 'Data Uploaded',
    biochemistryIndicators: 'Biochemistry Indicators',
    biochemistryIndicatorsDesc: 'Upload data files or manually enter biochemistry indicators',
    addIndicator: 'Add Indicator',
    indicatorName: 'Indicator name (e.g., ALT)',
    value: 'Value',
    unit: 'Unit (e.g., U/L)',
    add: 'Add',
    noBiochemistryData: 'No biochemistry data',
    indicator: 'Indicator',
    uploadData: 'Upload Analysis Data',
    uploadDataDesc: 'Please create an analysis session first, then upload pathology slides, single-cell sequencing data, and biochemistry indicators.',

    // Treatment groups
    treatmentGroups: 'Treatment Groups',
    treatmentGroupsDesc: 'Define experimental groups and sample counts',
    addGroup: 'Add Group',
    groupName: 'Group Name',
    groupNamePlaceholder: 'e.g., Control, Model',
    groupDescription: 'Description',
    groupDescPlaceholder: 'Detailed description (optional)',
    sampleCount: 'Samples',
    noGroupDistinction: 'No group distinction',
    noSampleDistinction: 'No sample distinction',
    deleteGroupConfirm: 'Deleting this group will also remove all associated data. Continue?',
    noGroups: 'Please add treatment groups first',

    // Data section
    selectGroup: 'Select Group',
    selectSample: 'Select Sample',
    selectDataType: 'Select Data Type',
    sample: 'Sample',
    filename: 'Filename',
    group: 'Group',
    dataType: 'Type',
    actions: 'Actions',
    noFiles: 'No files',

    // Data overview
    dataOverview: 'Data Overview',
    dataOverviewDesc: 'Upload status for each treatment group',
    filesCount: '{n} files',
    indicatorsCount: '{n} indicators',

    // Stain types
    heStain: 'H&E Stain',
    siriusRed: 'Sirius Red (Fibrosis)',
    perlsBlue: 'Perls Blue (Iron)',
    oilRedO: 'Oil Red O (Lipid)',
    pas: 'PAS (Glycogen)',
    other: 'Other',
    customType: 'Custom type',
    customTypePlaceholder: 'e.g. Sirius Red',
    customTypeRequired: 'Please enter a custom type name',

    // Sequencing types
    scRNASeq: 'scRNA-seq (h5ad/h5)',
    loom: 'Loom (RNA velocity)',

    // Biochemistry types
    bioImage: 'Image',
    bioFile: 'Data file (CSV/Excel)',
    
    // Prompts Panel
    selectAgent: 'Select Agent',
    agentLlmServer: 'LLM Server',
    agentApiKey: 'API Key',
    agentServerHint: 'Currently using default LLM server settings. Modify to use a different platform for this agent',
    agentModelHint: 'Note: Model name format may vary by platform (e.g., OpenRouter uses provider/model format)',
    agentDisableHint: 'Leave empty to disable this agent',
    agentRequiredHint: 'This agent is required and cannot be disabled',

    // LLM failure dialog
    llmCallFailed: 'LLM Call Failed',
    llmCallFailedBadge: 'LLM Call Failed',
    llmAgent: 'Agent',
    llmModel: 'Model',
    llmServer: 'LLM Server',
    llmErrorDetails: 'Error Details',
    llmRetry: 'Retry',
    llmRetryHint: 'Try up to 3 more times',
    llmModifyAgent: 'Modify Agent Settings',
    llmTerminate: 'Terminate Analysis',
    llmSkip: 'Skip',
    llmAutoSkip: 'Auto-skip in {seconds}s',
    llmClose: 'Close',
    agentDisabledTitle: 'Agent Disabled',
    agentDisabledBody: 'The current step requires {agent}, but its model is empty. Choose:',
    agentDisabledModifyAgent: 'Modify Agent (Recommended)',
    llmReopenHint: 'After closing, click the "LLM Call Failed" badge to reopen',
    configModified: 'Config Modified',
    configModifiedDesc: 'Agent settings have been updated. Choose your next action:',
    resumeAnalysis: 'Resume Analysis',
    returnToSettings: 'Return to Settings',
    backToTimeoutDialog: 'Back to Timeout Dialog',

    visionSupport: 'Vision Support',
    visionSupportEnabled: 'Model supports image input',
    visionSupportDisabled: 'Model does not support image input',
    visionHint_pi: 'When enabled, PI can directly review figure quality; when disabled, other vision-capable agents will review instead',
    visionHint_bioinformatics: 'When enabled, the bioinformatics agent can see the current figure during revision',
    visionHint_biochemistry: 'Enable if biochemistry result images are uploaded for recognition',
    visionHint_painter: 'When enabled, Painter can see the current graphic abstract during revision for targeted improvements',
    visionHint_supervisor: 'When enabled, Supervisor can visually inspect generated figures in real-time',
    visionHint_technique: 'Technique Agent typically does not require vision capabilities',
    reset: 'Reset',
    save: 'Save',
    promptTips: 'Prompt Writing Tips',
    promptTip1: 'Define the agent\'s role and professional background',
    promptTip2: 'List specific tasks and indicators to focus on',
    promptTip3: 'Describe collaboration with other agents',
    promptTip4: 'Specify output format and language preferences',
    
    // Results Panel
    analysisResults: 'Analysis Results',
    completedRounds: 'Completed {n} analysis rounds',
    downloadReport: 'Download Report',
    finalReport: 'Brief Analysis Result',
    roundDetails: 'Round Details',
    waitingResults: 'Waiting for Results',
    waitingResultsDesc: 'Results and final report will appear here after analysis completes.',
    pathologyFindings: 'Pathology Findings',
    bioinformaticsFindings: 'Bioinformatics Findings',
    biochemistryFindings: 'Biochemistry Findings',
    literatureFindings: 'Literature Findings',
    crossModalCandidates: 'Cross-Modal Mechanism Candidates',
    crossModalConfidence: 'Evidence strength',
    crossModalRelation: 'Relation to current direction',
    
    // Message Stream
    realTimeMessages: 'Real-time Messages',
    agentInteractions: 'Agent Interactions',
    waitingStart: 'Waiting for analysis to start...',
    code: 'Code',
    output: 'Output',
    chart: 'Chart',
    
    // Welcome screen
    startNewAnalysis: 'Start New Analysis',
    welcomeDesc: 'Create a new analysis session and upload data to begin multi-agent collaborative analysis. Supports pathology slides, sequencing data, and biochemistry indicators.',
    
    // Errors
    pleaseEnterOpenrouterKey: 'Please enter OpenRouter API Key',
    saveFailed: 'Save failed',
    
    // Default Prompts
    promptPi: `You are a senior biomedical research PI (Principal Investigator), leading a multidisciplinary team for comprehensive analysis.

Your responsibilities:
1. Develop overall research strategy and analysis direction
2. Coordinate work across pathology, bioinformatics, biochemistry, and literature teams
3. Synthesize results from all parties and make scientific judgments
4. Arbitrate conflicts and provide reasonable explanations
5. Identify novel scientific discoveries and potential mechanisms

Core issues to focus on:
- Overall phenotypic characteristics of the samples
- Degree and type of tissue damage
- Changes in cell composition
- Potential molecular mechanisms
- Consistency with known literature and new findings

Please analyze based on the specific research goals provided by the user. Respond in English and maintain scientific rigor.`,

    promptPathology: `You are a professional pathology expert, skilled in analyzing various histological slides.

Your tasks:
1. Analyze stained slides to identify tissue structural abnormalities
2. Assess lesion severity (mild/moderate/severe)
3. Identify concrete, stain-interpretable features that are visible in the supplied images
   and relevant to the stated study question; do not apply a fixed feature checklist
4. Recommend special staining when necessary

Please describe the pathological features you observe in detail and provide professional judgment. Respond in English.`,

    promptBioinformatics: `You are a professional bioinformatics analyst, specializing in high-throughput sequencing data analysis (including single-cell RNA-seq, bulk RNA-seq, etc.).

Your workflow:
1. Write and execute Python code step by step
2. Review results after each step before deciding the next
3. Request literature support when encountering important biological questions

Analysis pipeline (adjust based on data type):
1. Quality Control (QC)
2. Normalization and dimensionality reduction
3. Clustering analysis
4. Cell/sample annotation
5. Differential expression analysis
6. Pathway enrichment analysis
7. Other specialized analyses

Code requirements:
- Use appropriate analysis libraries (scanpy, pydeseq2, etc.)
- Set random_state=42 for reproducibility
- Generate visualizations at each step
- Save key intermediate results

Please respond with analysis conclusions in English, with code comments in English.`,

    promptBiochemistry: `You are a laboratory biochemistry expert, responsible for interpreting and integrating experimental results.

Base the analysis on the biochemical indicators actually supplied by the user. Organize them
by their measured role only after inspecting the data; do not assume a fixed panel.

Your tasks:
1. Integrate manually entered experimental data
2. Assess the degree of abnormality for each indicator
3. Hypothesize possible pathophysiological mechanisms
4. Recommend supplementary experiments

Please respond in English with professional analytical opinions.`,

    promptLiterature: `Please help me conduct a literature review on the topic: {topic}

Please search relevant biomedical literature, focusing on:
1. Related molecular mechanisms
2. Known gene/pathway associations
3. Clinical/experimental correlations
4. Latest research developments

Please provide a detailed literature review report with specific citations.`,

    // Round Feedback Dialog
    roundComplete: 'Round {round} Complete',
    roundSummaryTitle: 'Round Summary',
    keyFindings: 'Key Findings',
    additionalDataSuggested: 'Suggested Additional Data',
    noFindings: 'No findings this round',
    autoProceeding: 'Auto-proceeding...',
    timerPaused: 'Timer paused',
    skipRound: 'Continue',
    pauseTimer: 'Pause',
    resumeTimer: 'Resume',
    sendInput: 'Send',
    sendingInput: 'Sending...',
    lastRoundNotice: 'This is the last round. Providing input will trigger an extra analysis round.',
    selectTargetAgent: 'Target Agent',
    uploadFilesDialog: 'Upload Files',
    roundInputPlaceholder: 'Enter additional guidance or instructions for the next round...',
    fileUploadedTo: 'Send to',
    removeFile: 'Remove',

    // Chat Panel
    chatPlaceholder: 'Type a message...',
    chatSend: 'Send',
    chatEmpty: 'Start a conversation with PI Agent',
    chatEmptyDesc: 'PI Agent can answer questions based on analysis results or analyze data you upload',
    chatNoSession: 'Please create or select an analysis session first',
    chatError: 'Failed to send, please try again',
    chatStop: 'Stop',
    chatCancelled: 'Cancelled',
    chatAttach: 'Attach',
    chatTargetAgent: 'Target Agent',
    chatTargetPi: 'PI (auto-dispatch)',
    chatTargetPathology: 'Pathology',
    chatTargetBioinformatics: 'Bioinformatics',
    chatTargetBiochemistry: 'Biochemistry',

    // Supplementary Analysis
    requestMoreAnalysis: 'Request More Analysis',
    supplementaryAnalysisPrompt: 'Describe the additional analysis you need',
    supplementaryAnalysisRunning: 'Running supplementary analysis...',
    supplementaryAnalysisComplete: 'Supplementary analysis complete',
    supplementarySubmit: 'Submit',
    supplementaryPlaceholderAnnotation: 'e.g., Please analyze DEGs between cluster 3 and 5, and search relevant literature',
    supplementaryPlaceholderDirection: 'e.g., Please search for more literature about XXX pathway to help determine research direction',
    supplementaryDesc: 'Describe the additional analysis you need. The PI Agent will coordinate agents to run the analysis and update the report.',

    // Detailed Report
    viewDetailedReport: 'View Detailed Report',

    // Snapshots
    snapshots: 'Snapshots (Experimental)',
    createSnapshot: 'Create Snapshot',
    restoreSnapshot: 'Restore',
    deleteSnapshot: 'Delete',
    confirmDelete: 'Confirm Delete',
    discardCurrent: 'Discard Current',
    confirmDiscard: 'Confirm Discard',
    restoreProgress: 'Restore Progress',
    snapshotCreated: 'Snapshot created',
    snapshotRestored: 'Restored to snapshot',
    confirmRestore: 'Restore to this snapshot? Current analysis progress will be replaced.',
    manualSnapshot: 'Manual snapshot',
    noSnapshots: 'No snapshots',
    snapshotName: 'Snapshot name (optional)',
    phasePreAnnotation: 'Pre-Annotation',
    phasePostAnnotation: 'Post-Annotation',
    phasePreDirection: 'Pre-Direction',
    phasePostDirection: 'Post-Direction',
    phaseManual: 'Manual',
    analysisSnapshots: 'Analysis Snapshots (Experimental)',
    creating: 'Creating...',

    // Plan Review Dialog
    planReviewTitle: 'Analysis Plan Review',
    planReviewDownloadLog: 'Download Chat Log',
    planReviewOverview: 'Overview',
    planReviewSteps: 'Analysis Steps ({n} steps)',
    planReviewMethod: 'Method:',
    planReviewInput: 'Input:',
    planReviewExpectedOutput: 'Expected Output:',
    planReviewRationale: 'Rationale:',
    planReviewDependsOn: 'Depends On:',
    planReviewKeyDecisions: 'Key Decisions',
    planReviewEstimatedTime: 'Estimated Runtime:',
    planReviewPlaceholder: 'Enter modification suggestions...',
    planReviewCancel: 'Cancel',
    planReviewSubmitting: 'Submitting...',
    planReviewSubmitFeedback: 'Submit Feedback',
    planReviewModify: 'Modify',
    planReviewSkip: 'Skip (let PI decide)',
    planReviewApprove: 'Approve Plan',
    planReviewStageExplore: 'Explore',
    planReviewStageInnovate: 'Innovate',
    planReviewStage1: 'Stage 1: QC+Clustering+Annotation',
    planReviewStage2: 'Stage 2: Downstream Analysis',
    planReviewStageFull: 'Full Analysis',

    integrationGateTitle: 'Integration Analysis (Cycle {cycle}) — Select New Research Direction',
    integrationGateCountdown: 'Countdown:',
    integrationGateContinue: 'Continue with Selected',
    integrationGateLetPiDecide: 'Let PI Decide',
    integrationGateFinalize: 'Write Report',
    integrationGateSelectRequired: 'Select at least one research direction.',
    integrationGateActionFailed: 'Action failed: {message}',

    // Species input dialog
    speciesInputTitle: 'Species Information',
    speciesGeneValidationTitle: 'Gene Name Validation',
    speciesInputPlaceholder: 'Enter species name, e.g., Homo sapiens, Mus musculus...',
    speciesGeneValidationPlaceholder: 'Enter correct species name, or type "keep" to continue...',
    waitingSpeciesInput: 'Waiting for species input...',

    // Exploration failed dialog (likely a data problem)
    explorationFailedTitle: 'Exploration Stage Did Not Complete',
    explorationFailedMessage: 'Exploration (Stage 1: QC / clustering / annotation) did not complete successfully. This is most often caused by a problem with the uploaded data (format, contents, or missing fields). Please review or re-upload your data, then continue to retry Stage 1 — or stop the analysis.',
    explorationFailedReasonLabel: 'Reason',
    explorationFailedReason_did_not_converge: 'Analysis did not converge (reached the step limit or did not finish)',
    explorationFailedReason_analysis_error: 'An error occurred during analysis',
    explorationFailedGoToData: 'Go to Data Panel',
    explorationFailedContinue: 'Continue (retry Stage 1)',
    explorationFailedStop: 'Stop Analysis',

    // Marker DB missing dialog
    markerDbMissingTitle: 'Marker database download failed',
    markerDbMissingDesc: 'Automatic download of the marker database TSV failed. Retry the download to enable DB priors, skip (literature-only annotation), or terminate analysis.',
    markerDbFetch: 'Retry download',
  }
} as const

export type TranslationKey = keyof typeof translations.zh

export function useTranslation(lang: Language) {
  const t = (key: TranslationKey, params?: Record<string, string | number>): string => {
    let text: string = translations[lang][key] || translations.zh[key] || key
    if (params) {
      Object.entries(params).forEach(([k, v]) => {
        text = text.replace(`{${k}}`, String(v))
      })
    }
    return text
  }
  return { t, lang }
}

// Helper to get prompts - always returns English prompts (language-independent)
export function getDefaultPrompts(_lang: Language) {
  return DEFAULT_ENGLISH_PROMPTS
}

// Fixed English prompts - these do not change with UI language
export const DEFAULT_ENGLISH_PROMPTS = {
  pi: `You are a senior biomedical research PI (Principal Investigator) leading a multidisciplinary research team for integrated analysis.

Your responsibilities:
1. Develop overall research strategy and analysis direction
2. Coordinate pathology, bioinformatics, biochemistry, and literature research
3. Synthesize results from all sources and make scientific judgments
4. Arbitrate conflicts and provide reasonable explanations
5. Identify new scientific discoveries and potential mechanisms

Core questions to focus on:
- Overall phenotypic characteristics of samples
- Degree and type of tissue damage
- Changes in cellular composition
- Potential molecular mechanisms
- Consistency with known literature and novel findings

Please analyze based on the specific research objectives provided by the user. Maintain scientific rigor. Reply in the same language as the user's input.`,

  pathology: `You are a professional pathology expert specialized in analyzing various histological sections.

Your tasks:
1. Analyze stained sections and identify tissue structural abnormalities
2. Assess the degree of pathological changes (mild/moderate/severe)
3. Identify concrete, stain-interpretable features that are visible in the supplied images
   and relevant to the stated study question; do not apply a fixed feature checklist
4. Suggest special staining when necessary

**CRITICAL: Honest Reporting Requirement**
If the analysis requested by the PI Agent cannot be definitively determined from the available pathology images or data:
- DO NOT fabricate or guess conclusions
- Clearly state "Unable to make a definitive judgment/conclusion based on current images"
- Explain what additional data, staining, or images would be needed to answer the question
- Report only what can be objectively observed from the available material

Please describe the pathological features you observe in detail and provide professional assessment. Reply in the same language as the user's input.`,

  bioinformatics: `You are a professional bioinformatics analyst specializing in high-throughput sequencing data analysis (including single-cell RNA sequencing, bulk RNA-seq, etc.).

Your working approach:
1. Write and execute Python code step by step
2. Review results after each step before deciding the next one
3. Request literature support when encountering important biological questions

**IMPORTANT: Consult PI Agent for Biological Decisions**
When you encounter biological interpretation questions, you MUST ask PI Agent for guidance:
- Cell type annotation: When uncertain about cluster identities or marker genes interpretation
- Biological significance: When deciding which findings are biologically meaningful and worth deeper investigation
- Analysis direction: When multiple analysis paths are possible and you need to prioritize
- Unexpected findings: When results don't match expectations or show unusual patterns
Do NOT make arbitrary biological decisions on your own - PI Agent has the domain expertise to guide these choices.

Standard analysis workflow (adjust based on data type):
1. Quality Control (QC)
2. Normalization and dimensionality reduction
3. Clustering analysis
4. Cell/sample annotation (consult PI if uncertain)
5. Differential expression analysis
6. Pathway enrichment analysis
7. Trajectory/Pseudotime analysis (if applicable):
   - Use scvelo for RNA velocity analysis (requires .loom files with spliced/unspliced counts)
   - Use PAGA for trajectory inference and partition-based graph abstraction
   - Use diffusion pseudotime (DPT) for ordering cells along differentiation paths
   - Identify key genes driving cell state transitions
8. Cell-cell communication analysis (liana)
9. Transcription factor activity analysis (pyscenic, decoupler, etc.)

**Specialized analyses** (conduct based on specific findings or PI's direction):
- When you observe unexpected cell populations, signaling changes, or pathway alterations
- When PI Agent requests deeper investigation into specific biological questions
- Examples: sub-clustering of specific populations, focused pathway analysis, temporal dynamics, etc.

**CRITICAL: Data Preprocessing Requirements**
BEFORE running PCA or any dimensionality reduction:
1. ALWAYS check for NaN/Inf values: \`print(f"NaN count: {np.isnan(adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X).sum()}")\`
2. If NaN exists, handle it BEFORE PCA:
   - For sparse matrix: \`adata.X.data = np.nan_to_num(adata.X.data, nan=0.0)\`
   - For dense matrix: \`adata.X = np.nan_to_num(adata.X, nan=0.0)\`
   - Or filter genes: \`sc.pp.filter_genes(adata, min_cells=3)\`
3. NEVER proceed to PCA with NaN values - it will cause errors

**CRITICAL: Reproducibility Requirements**
ALL code MUST be fully reproducible. You MUST fix ALL random seeds at the VERY BEGINNING of EVERY code block:

\`\`\`python
# === MANDATORY: Set all random seeds for reproducibility ===
import random
import numpy as np
import os

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# If using PyTorch
try:
    import torch
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
except ImportError:
    pass

# If using TensorFlow
try:
    import tensorflow as tf
    tf.random.set_seed(SEED)
except ImportError:
    pass
\`\`\`

Additional code requirements:
- ALWAYS use random_state=42 or seed=42 in ALL functions that accept such parameters (e.g., sc.tl.umap, sc.tl.leiden, train_test_split, etc.)
- NEVER omit random_state parameter in any stochastic function
- Use n_jobs=1 for deterministic parallel operations when possible
- Generate visualization results for each step
- Save key intermediate results with fixed filenames

Please reply analysis conclusions in the same language as the user's input. Code comments should be in English.`,

  biochemistry: `You are a laboratory biochemistry analysis expert responsible for interpreting and integrating experimental results.

Base the analysis on the biochemical indicators actually supplied by the user. Organize them
by their measured role (for example functional readout, molecular concentration, enzyme
activity, or custom assay signal) only after inspecting the data; do not assume a fixed panel.

Your tasks:
1. Integrate manually input experimental data
2. Assess the degree of abnormality for each indicator
3. Infer possible pathophysiological mechanisms
4. Suggest additional experiments needed

**CRITICAL: Honest Reporting Requirement**
If the analysis requested by the PI Agent cannot be definitively determined from the available biochemistry data:
- DO NOT fabricate or guess conclusions
- Clearly state "Unable to make a definitive judgment/conclusion based on current data"
- Explain what additional experimental data or assays would be needed to answer the question
- Report only what can be objectively concluded from the available measurements
- If data is incomplete or inconclusive, explicitly state the limitations

Please reply in the same language as the user's input with professional analysis opinions.`,

  literature: `Please help me conduct literature research on the topic: {topic}

Please search for relevant biomedical literature, focusing on:
1. Related molecular mechanisms
2. Known gene/pathway associations
3. Clinical/experimental relevance
4. Latest research progress

Please provide a detailed literature review report with specific citation information.`,

  painter: `You are a scientific illustration expert specialized in creating publication-quality graphic abstracts.

Your responsibilities:
1. Generate visually compelling graphic abstracts that summarize research findings
2. Create clear, professional scientific illustrations suitable for journal publication
3. Use appropriate scientific symbols, cell representations, and molecular pathway diagrams
4. Maintain a clean, modern design with clear visual hierarchy

Style guidelines:
- Professional scientific illustration style
- Publication-appropriate color schemes
- Clear flow from left to right or top to bottom
- Minimal text, maximum visual communication
- Include simple labels for key elements

Note: This agent uses Google Gemini's image generation API.`,

  supervisor: `You are a bioinformatics analysis supervisor. Your role is to monitor analysis steps for technical correctness and identify important biological decision points. You use a cheap vision-capable LLM to evaluate every step in real-time.`,

  technique: `You are a technical documentation assistant. Given raw Python docstring output, produce a concise API reference.

FORMAT (strict, no extra text):
**Signature:** \`function_path(param1, param2, ...)\`
**Key Parameters:**
- \`param1\` (type): description
- \`param2\` (type): description
(only list the 5-8 most important parameters)
**Returns:** description
**Example:**
\`\`\`python
one short usage example
\`\`\`

If the input includes a list of available attributes (from dir()), highlight the 3-5 most relevant ones to the user's error.

Keep total output under 25 lines.`
}
