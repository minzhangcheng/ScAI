// Internationalization translations

export type Language = 'zh' | 'en'

export const translations = {
  zh: {
    // App
    appName: 'InteSc',
    appSubtitle: '智能单细胞测序整合分析',
    appFullName: 'InteSc - 智能单细胞测序整合分析',
    
    // Tabs
    tabAnalysis: '分析',
    tabData: '数据',
    tabPrompts: 'Agents',
    tabResults: '结果',
    tabReport: '报告',
    
    // Sidebar
    newAnalysis: '新建分析',
    exportState: '导出状态',
    importState: '导入状态',
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
    literatureGeminiNote: '需要配置 Google AI API Key',
    literatureOpenrouterDeepResearch: 'OpenRouter Deep Research',
    literatureOpenrouterDesc: '使用具有内置搜索功能的 AI 模型进行文献调研',
    literatureSearchModel: '搜索模型',
    literatureOpenrouterModelOptions: '推荐: perplexity/sonar-deep-research, openai/o4-mini-deep-research',
    literaturePubmedLlm: '仅检索 PubMed',
    literaturePubmedDesc: '仅使用 PubMed 数据库搜索文献 + LLM 综合分析',
    literaturePubmedEmailLabel: 'PubMed 邮箱',
    literaturePubmedEmailNote: 'NCBI E-utilities 要求提供联系邮箱',
    literatureGoogleApiKey: 'Google AI API Key',
    literatureMaxSearchRounds: '最大搜索轮数',
    literatureMaxSearchRoundsNote: 'Agent 可根据搜索结果动态调整查询，进行多轮迭代搜索',
    literaturePubmedSettings: 'PubMed 搜索设置',
    literaturePubmedSettingsNote: '所有模式在搜索失败时都会回退到 PubMed 检索',
    painterAgent: 'Painter Agent',
    painterAgentDesc: '图形摘要绘制',

    // Publication figures and reports
    figureGallery: '图片库',
    figureApproved: '已通过',
    figurePending: '待审核',
    figureRevisionRequested: '需修改',
    graphicAbstract: '图形摘要',
    integratedReport: '整合报告',
    downloadPDF: '下载 PDF',
    downloadPNG: '下载 PNG',
    downloadAllFigures: '下载所有图片',
    downloadAllTables: '下载所有表格',
    generatingGraphicAbstract: '正在生成图形摘要...',
    noFiguresYet: '暂无图片',
    noTablesYet: '暂无表格',
    piReviewingFigure: 'PI 正在审核图片...',
    figureApprovedSuccess: '图片已通过审核',
    figureNeedsRevision: '图片需要修改',
    qualityScore: '质量评分',
    
    // Data Panel
    pathologySlides: '病理切片',
    pathologySlidesDesc: '上传 H&E 或特殊染色切片',
    dragOrClick: '拖拽图片到这里，或点击选择',
    supportedFormats: '支持 PNG, JPEG, TIFF 格式',
    sequencingData: '单细胞测序数据',
    sequencingDataDesc: '上传 h5ad 或 10x Genomics h5 格式文件',
    uploadSequencing: '点击或拖拽上传测序数据',
    supportedH5ad: '支持 .h5ad (AnnData) 或 .h5 (10x Genomics)',
    dataUploaded: '数据已上传',
    biochemistryIndicators: '生化指标',
    biochemistryIndicatorsDesc: '输入实验测得的生化数据',
    addIndicator: '添加指标',
    indicatorName: '指标名称 (如 ALT)',
    value: '数值',
    unit: '单位 (如 U/L)',
    add: '添加',
    noBiochemistryData: '暂无生化数据',
    indicator: '指标',
    uploadData: '上传分析数据',
    uploadDataDesc: '请先创建一个分析会话，然后上传病理切片、单细胞测序数据和生化指标。',
    
    // Stain types
    heStain: 'H&E 染色',
    siriusRed: 'Sirius Red (纤维化)',
    perlsBlue: 'Perls Blue (铁沉积)',
    oilRedO: 'Oil Red O (脂肪)',
    pas: 'PAS (糖原)',
    other: '其他',
    
    // Prompts Panel
    selectAgent: '选择 Agent',
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
    promptPi: `你是一位资深的生物医学研究PI（Principal Investigator），负责领导一个多学科研究团队进行综合分析。

你的职责：
1. 制定整体研究策略和分析方向
2. 协调病理、生信、生化和文献各个方向的工作
3. 综合各方结果，做出科学判断
4. 当出现冲突时，进行裁决并给出合理解释
5. 识别新的科学发现和潜在机制

你需要关注的核心问题：
- 样本的整体表型特征
- 组织损伤的程度和类型
- 细胞组成的变化
- 潜在的分子机制
- 与已知文献的一致性和新发现

请根据用户提供的具体研究目标进行分析，用中文回复，保持科学严谨性。`,

    promptPathology: `你是一位专业的病理学专家，擅长分析各类组织学切片。

你的任务：
1. 分析染色切片，识别组织结构异常
2. 评估病变程度（轻度/中度/重度）
3. 识别特定病理特征：
   - 细胞形态学变化
   - 炎症浸润情况
   - 纤维化迹象
   - 特殊结构改变
   - 异常沉积物
4. 必要时建议进行特殊染色

请详细描述你观察到的病理特征，并给出专业判断。用中文回复。`,

    promptBioinformatics: `你是一位专业的生物信息学分析师，专注于高通量测序数据分析（包括单细胞RNA测序、bulk RNA-seq等）。

你的工作方式：
1. 一步一步编写和执行Python代码
2. 每一步都需要查看结果后再决定下一步
3. 遇到重要的生物学问题时，需要请求文献支持

分析流程（根据数据类型调整）：
1. 数据质控（QC）
2. 标准化和降维
3. 聚类分析
4. 细胞/样本注释
5. 差异表达分析
6. 通路富集分析
7. 其他专项分析

代码要求：
- 使用适当的分析库（scanpy、DESeq2等）
- 设置random_state=42确保可重复性
- 每一步都要生成可视化结果
- 保存关键中间结果

请用中文回复分析结论，代码用英文注释。`,

    promptBiochemistry: `你是一位实验室生化分析专家，负责解读和整合实验结果。

你需要关注各类生化指标，包括但不限于：
1. 组织/器官功能指标
2. 代谢相关指标
3. 氧化应激指标
4. 炎症指标
5. 其他专项检测

你的任务：
1. 整合人工输入的实验数据
2. 判断各指标的异常程度
3. 推测可能的病理生理机制
4. 建议需要补充的实验

请用中文回复，提供专业的分析意见。`,

    promptLiterature: `请帮助我进行文献调研，主题是：{topic}

请搜索相关的生物医学文献，重点关注：
1. 相关的分子机制
2. 已知的基因/通路关联
3. 临床/实验相关性
4. 最新的研究进展

请提供详细的文献综述报告，包含具体的引用信息。`,

    // Round Feedback Dialog
    roundComplete: '第 {round} 轮分析完成',
    roundSummaryTitle: '本轮总结',
    keyFindings: '主要发现',
    additionalDataSuggested: '建议补充的数据',
    noFindings: '本轮无发现',
    autoProceeding: '即将自动继续...',
    timerPaused: '计时已暂停',
    skipRound: '跳过',
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
  },

  en: {
    // App
    appName: 'InteSc',
    appSubtitle: 'Integrated & Intelligent Single Cell Analysis',
    appFullName: 'InteSc - Integrated & Intelligent Single Cell Analysis',
    
    // Tabs
    tabAnalysis: 'Analysis',
    tabData: 'Data',
    tabPrompts: 'Agents',
    tabResults: 'Results',
    tabReport: 'Report',
    
    // Sidebar
    newAnalysis: 'New Analysis',
    exportState: 'Export State',
    importState: 'Import State',
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
    literatureGeminiNote: 'Requires Google AI API Key',
    literatureOpenrouterDeepResearch: 'OpenRouter Deep Research',
    literatureOpenrouterDesc: 'Literature research using AI models with built-in web search',
    literatureSearchModel: 'Search Model',
    literatureOpenrouterModelOptions: 'Recommended: perplexity/sonar-deep-research, openai/o4-mini-deep-research',
    literaturePubmedLlm: 'PubMed Search Only',
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

    // Publication figures and reports
    figureGallery: 'Figure Gallery',
    figureApproved: 'Approved',
    figurePending: 'Pending Review',
    figureRevisionRequested: 'Revision Requested',
    graphicAbstract: 'Graphic Abstract',
    integratedReport: 'Integrated Report',
    downloadPDF: 'Download PDF',
    downloadPNG: 'Download PNG',
    downloadAllFigures: 'Download All Figures',
    downloadAllTables: 'Download All Tables',
    generatingGraphicAbstract: 'Generating graphic abstract...',
    noFiguresYet: 'No figures yet',
    noTablesYet: 'No tables yet',
    piReviewingFigure: 'PI reviewing figure...',
    figureApprovedSuccess: 'Figure approved',
    figureNeedsRevision: 'Figure needs revision',
    qualityScore: 'Quality Score',
    
    // Data Panel
    pathologySlides: 'Pathology Slides',
    pathologySlidesDesc: 'Upload H&E or special stain slides',
    dragOrClick: 'Drag images here, or click to select',
    supportedFormats: 'Supports PNG, JPEG, TIFF formats',
    sequencingData: 'Single-cell Sequencing Data',
    sequencingDataDesc: 'Upload h5ad or 10x Genomics h5 format file',
    uploadSequencing: 'Click or drag to upload sequencing data',
    supportedH5ad: 'Supports .h5ad (AnnData) or .h5 (10x Genomics)',
    dataUploaded: 'Data Uploaded',
    biochemistryIndicators: 'Biochemistry Indicators',
    biochemistryIndicatorsDesc: 'Enter experimental biochemistry data',
    addIndicator: 'Add Indicator',
    indicatorName: 'Indicator name (e.g., ALT)',
    value: 'Value',
    unit: 'Unit (e.g., U/L)',
    add: 'Add',
    noBiochemistryData: 'No biochemistry data',
    indicator: 'Indicator',
    uploadData: 'Upload Analysis Data',
    uploadDataDesc: 'Please create an analysis session first, then upload pathology slides, single-cell sequencing data, and biochemistry indicators.',
    
    // Stain types
    heStain: 'H&E Stain',
    siriusRed: 'Sirius Red (Fibrosis)',
    perlsBlue: 'Perls Blue (Iron)',
    oilRedO: 'Oil Red O (Lipid)',
    pas: 'PAS (Glycogen)',
    other: 'Other',
    
    // Prompts Panel
    selectAgent: 'Select Agent',
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
3. Identify specific pathological features:
   - Cell morphological changes
   - Inflammatory infiltration
   - Signs of fibrosis
   - Special structural changes
   - Abnormal deposits
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
- Use appropriate analysis libraries (scanpy, DESeq2, etc.)
- Set random_state=42 for reproducibility
- Generate visualizations at each step
- Save key intermediate results

Please respond with analysis conclusions in English, with code comments in English.`,

    promptBiochemistry: `You are a laboratory biochemistry expert, responsible for interpreting and integrating experimental results.

Focus on various biochemical indicators, including but not limited to:
1. Tissue/organ function indicators
2. Metabolism-related indicators
3. Oxidative stress indicators
4. Inflammatory markers
5. Other specialized tests

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
    skipRound: 'Skip',
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
3. Identify specific pathological features:
   - Cell morphological changes
   - Inflammatory infiltration
   - Signs of fibrosis
   - Special structural alterations
   - Abnormal deposits
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
   - Use scVelo for RNA velocity analysis (requires .loom files with spliced/unspliced counts)
   - Use PAGA for trajectory inference and partition-based graph abstraction
   - Use diffusion pseudotime (DPT) for ordering cells along differentiation paths
   - Identify key genes driving cell state transitions
8. Cell-cell communication analysis (CellChat, CellPhoneDB, etc.)
9. Transcription factor activity analysis (pySCENIC, decoupleR, etc.)

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

You need to focus on various biochemical indicators, including but not limited to:
1. Tissue/organ function indicators
2. Metabolism-related indicators
3. Oxidative stress indicators
4. Inflammatory indicators
5. Other specialized tests

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

Note: This agent uses Google Gemini's image generation API.`
}
