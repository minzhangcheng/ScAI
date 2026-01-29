import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { useAppStore } from '../stores/appStore'
import { useSession } from '../hooks/useApi'
import { useTranslation } from '../i18n/translations'
import { 
  Upload, 
  Image as ImageIcon, 
  FileSpreadsheet,
  FlaskConical,
  Plus,
  Check,
  File,
  Zap,
  Trash2,
  X
} from 'lucide-react'

export function DataPanel() {
  const { currentSession, language } = useAppStore()
  const { 
    uploadPathology, 
    uploadSnRNASeq, 
    uploadLoom, 
    addBiochemistry, 
    uploadBiochemistryFile,
    deletePathologyImage,
    deleteBiochemistryFile,
    deleteBiochemistryResult,
    deleteSequencing,
    deleteLoom
  } = useSession()
  const { t } = useTranslation(language)
  
  // Pathology states
  const [stainType, setStainType] = useState<'H&E' | 'Other'>('H&E')
  const [customStainType, setCustomStainType] = useState('')
  const [uploadingPathology, setUploadingPathology] = useState(false)
  const [pathologyProgress, setPathologyProgress] = useState(0)
  
  // Sequencing states
  const [uploadingControl, setUploadingControl] = useState(false)
  const [uploadingTreatment, setUploadingTreatment] = useState(false)
  const [controlProgress, setControlProgress] = useState(0)
  const [treatmentProgress, setTreatmentProgress] = useState(0)
  
  // Loom file states
  const [uploadingControlLoom, setUploadingControlLoom] = useState(false)
  const [uploadingTreatmentLoom, setUploadingTreatmentLoom] = useState(false)
  const [controlLoomProgress, setControlLoomProgress] = useState(0)
  const [treatmentLoomProgress, setTreatmentLoomProgress] = useState(0)
  
  // Biochemistry states
  const [showBioForm, setShowBioForm] = useState(false)
  const [bioIndicator, setBioIndicator] = useState('')
  const [bioValue, setBioValue] = useState('')
  const [bioUnit, setBioUnit] = useState('')
  const [uploadingBioFile, setUploadingBioFile] = useState(false)
  const [bioFileProgress, setBioFileProgress] = useState(0)

  // Get the actual stain type to use
  const getStainTypeValue = () => {
    if (stainType === 'H&E') return 'H&E'
    return customStainType || 'Other'
  }

  // Pathology dropzone with progress simulation
  const onDropPathology = useCallback(async (files: File[]) => {
    if (!currentSession || files.length === 0) return
    
    const finalStainType = getStainTypeValue()
    if (stainType === 'Other' && !customStainType) {
      alert(language === 'zh' ? '请输入染色类型' : 'Please enter stain type')
      return
    }
    
    setUploadingPathology(true)
    setPathologyProgress(0)
    
    try {
      for (let i = 0; i < files.length; i++) {
        const file = files[i]
        // Simulate progress for large files
        const progressInterval = setInterval(() => {
          setPathologyProgress(prev => Math.min(prev + 10, 90))
        }, 200)
        
        await uploadPathology(file, finalStainType)
        
        clearInterval(progressInterval)
        setPathologyProgress(((i + 1) / files.length) * 100)
      }
    } catch (e) {
      console.error('Upload failed:', e)
    } finally {
      setUploadingPathology(false)
      setPathologyProgress(0)
    }
  }, [currentSession, stainType, customStainType, uploadPathology, language])

  const { getRootProps: getPathologyRootProps, getInputProps: getPathologyInputProps, isDragActive: isPathologyDragActive } = useDropzone({
    onDrop: onDropPathology,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.tiff', '.tif']
    },
    disabled: uploadingPathology
  })

  // Control group dropzone
  const onDropControl = useCallback(async (files: File[]) => {
    if (!currentSession || files.length === 0) return
    
    setUploadingControl(true)
    setControlProgress(0)
    
    try {
      const progressInterval = setInterval(() => {
        setControlProgress(prev => Math.min(prev + 5, 90))
      }, 300)
      
      await uploadSnRNASeq(files[0], 'control')
      
      clearInterval(progressInterval)
      setControlProgress(100)
    } catch (e) {
      console.error('Upload failed:', e)
    } finally {
      setTimeout(() => {
        setUploadingControl(false)
        setControlProgress(0)
      }, 500)
    }
  }, [currentSession, uploadSnRNASeq])

  // Treatment group dropzone
  const onDropTreatment = useCallback(async (files: File[]) => {
    if (!currentSession || files.length === 0) return
    
    setUploadingTreatment(true)
    setTreatmentProgress(0)
    
    try {
      const progressInterval = setInterval(() => {
        setTreatmentProgress(prev => Math.min(prev + 5, 90))
      }, 300)
      
      await uploadSnRNASeq(files[0], 'treatment')
      
      clearInterval(progressInterval)
      setTreatmentProgress(100)
    } catch (e) {
      console.error('Upload failed:', e)
    } finally {
      setTimeout(() => {
        setUploadingTreatment(false)
        setTreatmentProgress(0)
      }, 500)
    }
  }, [currentSession, uploadSnRNASeq])

  const { getRootProps: getControlRootProps, getInputProps: getControlInputProps, isDragActive: isControlDragActive } = useDropzone({
    onDrop: onDropControl,
    accept: {
      'application/octet-stream': ['.h5ad', '.h5'],
      'application/x-hdf5': ['.h5ad', '.h5'],
      'application/gzip': ['.h5ad.gz']
    },
    multiple: false,
    disabled: uploadingControl
  })

  const { getRootProps: getTreatmentRootProps, getInputProps: getTreatmentInputProps, isDragActive: isTreatmentDragActive } = useDropzone({
    onDrop: onDropTreatment,
    accept: {
      'application/octet-stream': ['.h5ad', '.h5'],
      'application/x-hdf5': ['.h5ad', '.h5'],
      'application/gzip': ['.h5ad.gz']
    },
    multiple: false,
    disabled: uploadingTreatment
  })

  // Control group loom dropzone
  const onDropControlLoom = useCallback(async (files: File[]) => {
    if (!currentSession || files.length === 0) return
    
    setUploadingControlLoom(true)
    setControlLoomProgress(0)
    
    try {
      const progressInterval = setInterval(() => {
        setControlLoomProgress(prev => Math.min(prev + 5, 90))
      }, 300)
      
      await uploadLoom(files[0], 'control')
      
      clearInterval(progressInterval)
      setControlLoomProgress(100)
    } catch (e) {
      console.error('Upload failed:', e)
    } finally {
      setTimeout(() => {
        setUploadingControlLoom(false)
        setControlLoomProgress(0)
      }, 500)
    }
  }, [currentSession, uploadLoom])

  // Treatment group loom dropzone
  const onDropTreatmentLoom = useCallback(async (files: File[]) => {
    if (!currentSession || files.length === 0) return
    
    setUploadingTreatmentLoom(true)
    setTreatmentLoomProgress(0)
    
    try {
      const progressInterval = setInterval(() => {
        setTreatmentLoomProgress(prev => Math.min(prev + 5, 90))
      }, 300)
      
      await uploadLoom(files[0], 'treatment')
      
      clearInterval(progressInterval)
      setTreatmentLoomProgress(100)
    } catch (e) {
      console.error('Upload failed:', e)
    } finally {
      setTimeout(() => {
        setUploadingTreatmentLoom(false)
        setTreatmentLoomProgress(0)
      }, 500)
    }
  }, [currentSession, uploadLoom])

  const { getRootProps: getControlLoomRootProps, getInputProps: getControlLoomInputProps, isDragActive: isControlLoomDragActive } = useDropzone({
    onDrop: onDropControlLoom,
    accept: {
      'application/octet-stream': ['.loom'],
    },
    multiple: false,
    disabled: uploadingControlLoom
  })

  const { getRootProps: getTreatmentLoomRootProps, getInputProps: getTreatmentLoomInputProps, isDragActive: isTreatmentLoomDragActive } = useDropzone({
    onDrop: onDropTreatmentLoom,
    accept: {
      'application/octet-stream': ['.loom'],
    },
    multiple: false,
    disabled: uploadingTreatmentLoom
  })

  const handleAddBiochemistry = async () => {
    if (!bioIndicator || !bioValue || !bioUnit) return
    
    try {
      await addBiochemistry(bioIndicator, parseFloat(bioValue), bioUnit)
      setBioIndicator('')
      setBioValue('')
      setBioUnit('')
      setShowBioForm(false)
    } catch (e) {
      console.error('Failed to add biochemistry:', e)
    }
  }

  // Biochemistry file upload handler
  const onDropBioFile = useCallback(async (files: File[]) => {
    if (!currentSession || files.length === 0) return
    
    setUploadingBioFile(true)
    setBioFileProgress(0)
    
    try {
      for (let i = 0; i < files.length; i++) {
        const file = files[i]
        const progressInterval = setInterval(() => {
          setBioFileProgress(prev => Math.min(prev + 10, 90))
        }, 200)
        
        await uploadBiochemistryFile(file)
        
        clearInterval(progressInterval)
        setBioFileProgress(((i + 1) / files.length) * 100)
      }
    } catch (e) {
      console.error('Failed to upload biochemistry file:', e)
    } finally {
      setUploadingBioFile(false)
      setBioFileProgress(0)
    }
  }, [currentSession, uploadBiochemistryFile])

  const { getRootProps: getBioFileRootProps, getInputProps: getBioFileInputProps, isDragActive: isBioFileDragActive } = useDropzone({
    onDrop: onDropBioFile,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'],
      'text/*': ['.csv', '.txt', '.tsv'],
      'application/vnd.ms-excel': ['.xls'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx']
    },
    multiple: true,
    disabled: uploadingBioFile
  })

  if (!currentSession) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-center">
        <div className="p-6 bg-slate-100 rounded-full mb-6">
          <Upload className="w-12 h-12 text-slate-400" />
        </div>
        <h2 className="text-xl font-semibold text-slate-900 mb-2">
          {t('uploadData')}
        </h2>
        <p className="text-slate-500 max-w-md">
          {t('uploadDataDesc')}
        </p>
      </div>
    )
  }

  return (
    <div className="space-y-8">
      {/* Pathology Images */}
      <section>
        <div className="mb-4">
          <h3 className="text-lg font-semibold text-slate-900">{t('pathologySlides')}</h3>
          <p className="text-sm text-slate-500">{t('pathologySlidesDesc')}</p>
        </div>

        {/* Stain Type Selection */}
        <div className="flex items-center gap-4 mb-4">
          <div className="flex items-center gap-2">
            <label className="text-sm text-slate-600">
              {language === 'zh' ? '染色类型:' : 'Stain Type:'}
            </label>
            <select
              value={stainType}
              onChange={(e) => setStainType(e.target.value as 'H&E' | 'Other')}
              className="px-3 py-2 border border-slate-300 rounded-lg text-sm"
            >
              <option value="H&E">H&E</option>
              <option value="Other">{language === 'zh' ? '其他' : 'Other'}</option>
            </select>
          </div>
          
          {stainType === 'Other' && (
            <input
              type="text"
              value={customStainType}
              onChange={(e) => setCustomStainType(e.target.value)}
              placeholder={language === 'zh' ? '输入染色类型 (如 Sirius Red)' : 'Enter stain type (e.g., Sirius Red)'}
              className="flex-1 px-3 py-2 border border-slate-300 rounded-lg text-sm"
            />
          )}
        </div>

        {/* Upload Area */}
        <div
          {...getPathologyRootProps()}
          className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all ${
            uploadingPathology 
              ? 'border-violet-300 bg-violet-50 cursor-not-allowed'
              : isPathologyDragActive
                ? 'border-violet-500 bg-violet-50'
                : 'border-slate-300 hover:border-violet-400 hover:bg-slate-50'
          }`}
        >
          <input {...getPathologyInputProps()} />
          <ImageIcon className="w-8 h-8 text-slate-400 mx-auto mb-2" />
          <p className="text-slate-600 font-medium text-sm">
            {uploadingPathology 
              ? (language === 'zh' ? '上传中...' : 'Uploading...')
              : isPathologyDragActive 
                ? (language === 'zh' ? '释放以上传' : 'Release to upload') 
                : t('dragOrClick')}
          </p>
          <p className="text-xs text-slate-400 mt-1">
            {t('supportedFormats')}
          </p>
        </div>

        {/* Upload Progress */}
        {uploadingPathology && (
          <div className="mt-3">
            <div className="flex items-center justify-between text-sm text-slate-600 mb-1">
              <span>{language === 'zh' ? '上传进度' : 'Upload Progress'}</span>
              <span>{Math.round(pathologyProgress)}%</span>
            </div>
            <div className="w-full bg-slate-200 rounded-full h-2">
              <div 
                className="bg-violet-600 h-2 rounded-full transition-all duration-300"
                style={{ width: `${pathologyProgress}%` }}
              />
            </div>
          </div>
        )}

        {/* Uploaded Files List */}
        {currentSession.pathologyImages.length > 0 && (
          <div className="mt-4 border border-slate-200 rounded-xl overflow-hidden">
            <table className="w-full">
              <thead className="bg-slate-50">
                <tr>
                  <th className="px-4 py-2 text-left text-sm font-medium text-slate-600">
                    {language === 'zh' ? '文件名' : 'Filename'}
                  </th>
                  <th className="px-4 py-2 text-left text-sm font-medium text-slate-600">
                    {language === 'zh' ? '染色类型' : 'Stain Type'}
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {currentSession.pathologyImages.map((img) => (
                  <tr key={img.id} className="hover:bg-slate-50">
                    <td className="px-4 py-2 text-sm text-slate-900">
                      <div className="flex items-center gap-2">
                        <File className="w-4 h-4 text-slate-400" />
                        {img.filename}
                      </div>
                    </td>
                    <td className="px-4 py-2 text-sm text-slate-600">
                      <div className="flex items-center justify-between">
                        <span className="px-2 py-1 bg-violet-100 text-violet-700 rounded text-xs font-medium">
                          {img.imageType}
                        </span>
                        <button
                          onClick={() => deletePathologyImage(img.id)}
                          className="p-1 text-slate-400 hover:text-red-500 hover:bg-red-50 rounded transition-colors"
                          title={language === 'zh' ? '删除' : 'Delete'}
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {/* snRNA-seq Data */}
      <section>
        <div className="mb-4">
          <h3 className="text-lg font-semibold text-slate-900">{t('sequencingData')}</h3>
          <p className="text-sm text-slate-500">{t('sequencingDataDesc')}</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Control Group */}
          <div>
            <h4 className="text-sm font-medium text-slate-700 mb-2 flex items-center gap-2">
              <span className="w-3 h-3 bg-blue-500 rounded-full"></span>
              {language === 'zh' ? '对照组 (Control)' : 'Control Group'}
            </h4>
            
            {currentSession.controlSeqPath ? (
              <div className="flex items-center gap-3 p-4 bg-blue-50 border border-blue-200 rounded-xl">
                <Check className="w-5 h-5 text-blue-600" />
                <div className="flex-1 min-w-0">
                  <p className="font-medium text-blue-900 text-sm">{t('dataUploaded')}</p>
                  <p className="text-xs text-blue-600 truncate">
                    {currentSession.controlSeqPath.split('/').pop()}
                  </p>
                </div>
                <button
                  onClick={() => deleteSequencing('control')}
                  className="p-1.5 text-blue-400 hover:text-red-500 hover:bg-red-50 rounded transition-colors"
                  title={language === 'zh' ? '删除' : 'Delete'}
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            ) : (
              <>
                <div
                  {...getControlRootProps()}
                  className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all ${
                    uploadingControl
                      ? 'border-blue-300 bg-blue-50 cursor-not-allowed'
                      : isControlDragActive
                        ? 'border-blue-500 bg-blue-50'
                        : 'border-slate-300 hover:border-blue-400 hover:bg-slate-50'
                  }`}
                >
                  <input {...getControlInputProps()} />
                  <FileSpreadsheet className="w-8 h-8 text-slate-400 mx-auto mb-2" />
                  <p className="text-slate-600 font-medium text-sm">
                    {uploadingControl 
                      ? (language === 'zh' ? '上传中...' : 'Uploading...')
                      : (language === 'zh' ? '上传对照组数据' : 'Upload control data')}
                  </p>
                  <p className="text-xs text-slate-400 mt-1">
                    .h5ad / .h5
                  </p>
                </div>
                
                {uploadingControl && (
                  <div className="mt-2">
                    <div className="w-full bg-slate-200 rounded-full h-1.5">
                      <div 
                        className="bg-blue-600 h-1.5 rounded-full transition-all duration-300"
                        style={{ width: `${controlProgress}%` }}
                      />
                    </div>
                  </div>
                )}
              </>
            )}
          </div>

          {/* Treatment Group */}
          <div>
            <h4 className="text-sm font-medium text-slate-700 mb-2 flex items-center gap-2">
              <span className="w-3 h-3 bg-orange-500 rounded-full"></span>
              {language === 'zh' ? '实验组 (Treatment)' : 'Treatment Group'}
            </h4>
            
            {currentSession.treatmentSeqPath ? (
              <div className="flex items-center gap-3 p-4 bg-orange-50 border border-orange-200 rounded-xl">
                <Check className="w-5 h-5 text-orange-600" />
                <div className="flex-1 min-w-0">
                  <p className="font-medium text-orange-900 text-sm">{t('dataUploaded')}</p>
                  <p className="text-xs text-orange-600 truncate">
                    {currentSession.treatmentSeqPath.split('/').pop()}
                  </p>
                </div>
                <button
                  onClick={() => deleteSequencing('treatment')}
                  className="p-1.5 text-orange-400 hover:text-red-500 hover:bg-red-50 rounded transition-colors"
                  title={language === 'zh' ? '删除' : 'Delete'}
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            ) : (
              <>
                <div
                  {...getTreatmentRootProps()}
                  className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all ${
                    uploadingTreatment
                      ? 'border-orange-300 bg-orange-50 cursor-not-allowed'
                      : isTreatmentDragActive
                        ? 'border-orange-500 bg-orange-50'
                        : 'border-slate-300 hover:border-orange-400 hover:bg-slate-50'
                  }`}
                >
                  <input {...getTreatmentInputProps()} />
                  <FileSpreadsheet className="w-8 h-8 text-slate-400 mx-auto mb-2" />
                  <p className="text-slate-600 font-medium text-sm">
                    {uploadingTreatment 
                      ? (language === 'zh' ? '上传中...' : 'Uploading...')
                      : (language === 'zh' ? '上传实验组数据' : 'Upload treatment data')}
                  </p>
                  <p className="text-xs text-slate-400 mt-1">
                    .h5ad / .h5
                  </p>
                </div>
                
                {uploadingTreatment && (
                  <div className="mt-2">
                    <div className="w-full bg-slate-200 rounded-full h-1.5">
                      <div 
                        className="bg-orange-600 h-1.5 rounded-full transition-all duration-300"
                        style={{ width: `${treatmentProgress}%` }}
                      />
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        </div>

        <p className="text-xs text-slate-400 mt-3">
          {t('supportedH5ad')}
        </p>
      </section>

      {/* RNA Velocity - Loom Files */}
      <section>
        <div className="mb-4">
          <h3 className="text-lg font-semibold text-slate-900 flex items-center gap-2">
            <Zap className="w-5 h-5 text-purple-500" />
            {language === 'zh' ? 'RNA 速率分析 (Loom 文件)' : 'RNA Velocity (Loom Files)'}
          </h3>
          <p className="text-sm text-slate-500">
            {language === 'zh' 
              ? '上传 velocyto 生成的 loom 文件，用于 RNA 速率分析 (可选)'
              : 'Upload loom files generated by velocyto for RNA velocity analysis (optional)'}
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Control Group Loom */}
          <div>
            <h4 className="text-sm font-medium text-slate-700 mb-2 flex items-center gap-2">
              <span className="w-3 h-3 bg-purple-400 rounded-full"></span>
              {language === 'zh' ? '对照组 Loom' : 'Control Loom'}
            </h4>
            
            {currentSession.controlLoomPath ? (
              <div className="flex items-center gap-3 p-4 bg-purple-50 border border-purple-200 rounded-xl">
                <Check className="w-5 h-5 text-purple-600" />
                <div className="flex-1 min-w-0">
                  <p className="font-medium text-purple-900 text-sm">{t('dataUploaded')}</p>
                  <p className="text-xs text-purple-600 truncate">
                    {currentSession.controlLoomPath.split('/').pop()}
                  </p>
                </div>
                <button
                  onClick={() => deleteLoom('control')}
                  className="p-1.5 text-purple-400 hover:text-red-500 hover:bg-red-50 rounded transition-colors"
                  title={language === 'zh' ? '删除' : 'Delete'}
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            ) : (
              <>
                <div
                  {...getControlLoomRootProps()}
                  className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all ${
                    uploadingControlLoom
                      ? 'border-purple-300 bg-purple-50 cursor-not-allowed'
                      : isControlLoomDragActive
                        ? 'border-purple-500 bg-purple-50'
                        : 'border-slate-300 hover:border-purple-400 hover:bg-slate-50'
                  }`}
                >
                  <input {...getControlLoomInputProps()} />
                  <Zap className="w-8 h-8 text-slate-400 mx-auto mb-2" />
                  <p className="text-slate-600 font-medium text-sm">
                    {uploadingControlLoom 
                      ? (language === 'zh' ? '上传中...' : 'Uploading...')
                      : (language === 'zh' ? '上传对照组 Loom' : 'Upload control loom')}
                  </p>
                  <p className="text-xs text-slate-400 mt-1">
                    .loom
                  </p>
                </div>
                
                {uploadingControlLoom && (
                  <div className="mt-2">
                    <div className="w-full bg-slate-200 rounded-full h-1.5">
                      <div 
                        className="bg-purple-600 h-1.5 rounded-full transition-all duration-300"
                        style={{ width: `${controlLoomProgress}%` }}
                      />
                    </div>
                  </div>
                )}
              </>
            )}
          </div>

          {/* Treatment Group Loom */}
          <div>
            <h4 className="text-sm font-medium text-slate-700 mb-2 flex items-center gap-2">
              <span className="w-3 h-3 bg-pink-500 rounded-full"></span>
              {language === 'zh' ? '实验组 Loom' : 'Treatment Loom'}
            </h4>
            
            {currentSession.treatmentLoomPath ? (
              <div className="flex items-center gap-3 p-4 bg-pink-50 border border-pink-200 rounded-xl">
                <Check className="w-5 h-5 text-pink-600" />
                <div className="flex-1 min-w-0">
                  <p className="font-medium text-pink-900 text-sm">{t('dataUploaded')}</p>
                  <p className="text-xs text-pink-600 truncate">
                    {currentSession.treatmentLoomPath.split('/').pop()}
                  </p>
                </div>
                <button
                  onClick={() => deleteLoom('treatment')}
                  className="p-1.5 text-pink-400 hover:text-red-500 hover:bg-red-50 rounded transition-colors"
                  title={language === 'zh' ? '删除' : 'Delete'}
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            ) : (
              <>
                <div
                  {...getTreatmentLoomRootProps()}
                  className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all ${
                    uploadingTreatmentLoom
                      ? 'border-pink-300 bg-pink-50 cursor-not-allowed'
                      : isTreatmentLoomDragActive
                        ? 'border-pink-500 bg-pink-50'
                        : 'border-slate-300 hover:border-pink-400 hover:bg-slate-50'
                  }`}
                >
                  <input {...getTreatmentLoomInputProps()} />
                  <Zap className="w-8 h-8 text-slate-400 mx-auto mb-2" />
                  <p className="text-slate-600 font-medium text-sm">
                    {uploadingTreatmentLoom 
                      ? (language === 'zh' ? '上传中...' : 'Uploading...')
                      : (language === 'zh' ? '上传实验组 Loom' : 'Upload treatment loom')}
                  </p>
                  <p className="text-xs text-slate-400 mt-1">
                    .loom
                  </p>
                </div>
                
                {uploadingTreatmentLoom && (
                  <div className="mt-2">
                    <div className="w-full bg-slate-200 rounded-full h-1.5">
                      <div 
                        className="bg-pink-600 h-1.5 rounded-full transition-all duration-300"
                        style={{ width: `${treatmentLoomProgress}%` }}
                      />
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        </div>

        <p className="text-xs text-slate-400 mt-3">
          {language === 'zh' 
            ? '支持 velocyto 或 loompy 生成的 .loom 格式文件，用于 scVelo RNA 速率分析'
            : 'Supports .loom files from velocyto or loompy for scVelo RNA velocity analysis'}
        </p>
      </section>

      {/* Biochemistry Results */}
      <section>
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-lg font-semibold text-slate-900">{t('biochemistryIndicators')}</h3>
            <p className="text-sm text-slate-500">{t('biochemistryIndicatorsDesc')}</p>
          </div>
          
          <button
            onClick={() => setShowBioForm(true)}
            className="flex items-center gap-2 px-3 py-2 bg-amber-100 text-amber-700 rounded-lg text-sm font-medium hover:bg-amber-200 transition-colors"
          >
            <Plus className="w-4 h-4" />
            {t('addIndicator')}
          </button>
        </div>

        {/* File/Image Upload */}
        <div
          {...getBioFileRootProps()}
          className={`
            mb-4 p-6 border-2 border-dashed rounded-xl transition-all cursor-pointer
            ${isBioFileDragActive 
              ? 'border-amber-500 bg-amber-50' 
              : 'border-slate-300 hover:border-amber-400 hover:bg-amber-50/50'}
            ${uploadingBioFile ? 'opacity-50 cursor-not-allowed' : ''}
          `}
        >
          <input {...getBioFileInputProps()} />
          <div className="text-center">
            {uploadingBioFile ? (
              <>
                <div className="w-full h-2 bg-slate-200 rounded-full overflow-hidden mb-2">
                  <div 
                    className="h-full bg-amber-500 transition-all"
                    style={{ width: `${bioFileProgress}%` }}
                  />
                </div>
                <p className="text-sm text-slate-600">{language === 'zh' ? '上传中...' : 'Uploading...'}</p>
              </>
            ) : (
              <>
                <div className="flex justify-center gap-2 mb-2">
                  <ImageIcon className="w-6 h-6 text-amber-400" />
                  <FileSpreadsheet className="w-6 h-6 text-amber-400" />
                </div>
                <p className="text-sm text-slate-600">
                  {language === 'zh' 
                    ? '拖拽上传生化数据图片或文件 (Excel, CSV, 图片)' 
                    : 'Drop biochemistry images or files here (Excel, CSV, images)'}
                </p>
                <p className="text-xs text-slate-400 mt-1">
                  {language === 'zh'
                    ? '支持实验结果截图、数据表格等'
                    : 'Supports screenshots of results, data tables, etc.'}
                </p>
              </>
            )}
          </div>
        </div>

        {/* Uploaded Files List */}
        {currentSession.biochemistryFiles && currentSession.biochemistryFiles.length > 0 && (
          <div className="mb-4 p-3 bg-amber-50 rounded-xl">
            <p className="text-sm font-medium text-amber-700 mb-2">
              {language === 'zh' ? '已上传文件' : 'Uploaded Files'}
            </p>
            <div className="space-y-1">
              {currentSession.biochemistryFiles.map((file) => (
                <div key={file.id} className="flex items-center gap-2 text-sm text-amber-600">
                  {file.fileType === 'image' ? (
                    <ImageIcon className="w-4 h-4" />
                  ) : (
                    <FileSpreadsheet className="w-4 h-4" />
                  )}
                  <span className="flex-1">{file.filename}</span>
                  <Check className="w-4 h-4 text-green-500" />
                  <button
                    onClick={() => deleteBiochemistryFile(file.id)}
                    className="p-1 text-amber-400 hover:text-red-500 hover:bg-red-50 rounded transition-colors"
                    title={language === 'zh' ? '删除' : 'Delete'}
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Add Form */}
        {showBioForm && (
          <div className="p-4 bg-slate-50 rounded-xl mb-4">
            <div className="grid grid-cols-3 gap-3">
              <input
                type="text"
                value={bioIndicator}
                onChange={(e) => setBioIndicator(e.target.value)}
                placeholder={t('indicatorName')}
                className="px-3 py-2 border border-slate-300 rounded-lg text-sm"
              />
              <input
                type="number"
                value={bioValue}
                onChange={(e) => setBioValue(e.target.value)}
                placeholder={t('value')}
                className="px-3 py-2 border border-slate-300 rounded-lg text-sm"
              />
              <input
                type="text"
                value={bioUnit}
                onChange={(e) => setBioUnit(e.target.value)}
                placeholder={t('unit')}
                className="px-3 py-2 border border-slate-300 rounded-lg text-sm"
              />
            </div>
            <div className="flex justify-end gap-2 mt-3">
              <button
                onClick={() => setShowBioForm(false)}
                className="px-3 py-1.5 text-sm text-slate-600 hover:bg-slate-200 rounded-lg"
              >
                {t('cancel')}
              </button>
              <button
                onClick={handleAddBiochemistry}
                disabled={!bioIndicator || !bioValue || !bioUnit}
                className="px-3 py-1.5 text-sm bg-amber-600 text-white rounded-lg hover:bg-amber-700 disabled:opacity-50"
              >
                {t('add')}
              </button>
            </div>
          </div>
        )}

        {/* Results Table */}
        {currentSession.biochemistryResults.length > 0 ? (
          <div className="bg-white border border-slate-200 rounded-xl overflow-hidden">
            <table className="w-full">
              <thead className="bg-slate-50">
                <tr>
                  <th className="px-4 py-3 text-left text-sm font-medium text-slate-600">{t('indicator')}</th>
                  <th className="px-4 py-3 text-left text-sm font-medium text-slate-600">{t('value')}</th>
                  <th className="px-4 py-3 text-left text-sm font-medium text-slate-600">{t('unit')}</th>
                  <th className="px-4 py-3 text-right text-sm font-medium text-slate-600 w-16"></th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {currentSession.biochemistryResults.map((result) => (
                  <tr key={result.id} className="hover:bg-slate-50">
                    <td className="px-4 py-3 text-sm text-slate-900 font-medium">
                      {result.indicator}
                    </td>
                    <td className="px-4 py-3 text-sm text-slate-600">
                      {result.value}
                    </td>
                    <td className="px-4 py-3 text-sm text-slate-500">
                      {result.unit}
                    </td>
                    <td className="px-4 py-3 text-right">
                      <button
                        onClick={() => deleteBiochemistryResult(result.id)}
                        className="p-1 text-slate-400 hover:text-red-500 hover:bg-red-50 rounded transition-colors"
                        title={language === 'zh' ? '删除' : 'Delete'}
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : !currentSession.biochemistryFiles?.length && (
          <div className="text-center py-8 text-slate-400">
            <FlaskConical className="w-10 h-10 mx-auto mb-2 opacity-50" />
            <p>{t('noBiochemistryData')}</p>
          </div>
        )}
      </section>
    </div>
  )
}
