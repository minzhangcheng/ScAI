// Data tab: composes the group manager, upload sections, and overview.
// Split out of frontend/src/components/DataPanel.tsx verbatim.

import { useAppStore } from '../../stores/appStore'
import { useSession } from '../../hooks/useApi'
import { useTranslation } from '../../i18n/translations'
import { Upload, FlaskConical, Microscope, Dna } from 'lucide-react'
import { PATHOLOGY_TYPES, BIOCHEMISTRY_TYPES } from '../../utils/dataTypes'
import { DataOverviewMatrix } from './DataOverviewMatrix'
import { DataSection } from './DataSection'
import { GroupManager } from './GroupManager'
import { IndicatorForm } from './IndicatorForm'
import { SEQUENCING_TYPES } from './uploadRules'

// ============================================================
// Main DataPanel Component
// ============================================================
export function DataPanel() {
  const { currentSession, language } = useAppStore()
  const {
    addGroup,
    updateGroup,
    deleteGroup,
    uploadData,
    deleteDataFile,
    addIndicator,
    deleteIndicator,
  } = useSession()
  const { t } = useTranslation(language)

  if (!currentSession) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-center">
        <div className="p-6 bg-slate-100 rounded-full mb-6">
          <Upload className="w-12 h-12 text-slate-400" />
        </div>
        <h2 className="font-display text-xl font-semibold text-slate-900 mb-2">
          {t('uploadData')}
        </h2>
        <p className="text-slate-500 max-w-md">
          {t('uploadDataDesc')}
        </p>
      </div>
    )
  }

  const groups = currentSession.treatmentGroups || []
  const dataFiles = currentSession.dataFiles || []
  const indicators = currentSession.biochemistryIndicators || []

  return (
    <div className="space-y-6">
      {/* Treatment Groups */}
      <GroupManager
        groups={groups}
        onAdd={addGroup}
        onUpdate={updateGroup}
        onDelete={deleteGroup}
        language={language}
        t={t}
      />

      {/* Pathology Section */}
      <DataSection
        section="pathology"
        title={t('pathologySlides')}
        description={t('pathologySlidesDesc')}
        icon={Microscope}
        iconColor="text-violet-500"
        groups={groups}
        dataFiles={dataFiles}
        typeOptions={PATHOLOGY_TYPES}
        multipleFiles={true}
        onUpload={uploadData}
        onDelete={deleteDataFile}
        language={language}
        t={t}
      />

      {/* Sequencing Section */}
      <DataSection
        section="sequencing"
        title={t('sequencingData')}
        description={t('sequencingDataDesc')}
        icon={Dna}
        iconColor="text-blue-500"
        groups={groups}
        dataFiles={dataFiles}
        typeOptions={SEQUENCING_TYPES}
        multipleFiles={true}
        onUpload={uploadData}
        onDelete={deleteDataFile}
        language={language}
        t={t}
      />

      {/* Biochemistry Section */}
      <DataSection
        section="biochemistry"
        title={t('biochemistryIndicators')}
        description={t('biochemistryIndicatorsDesc')}
        icon={FlaskConical}
        iconColor="text-amber-500"
        groups={groups}
        dataFiles={dataFiles}
        typeOptions={BIOCHEMISTRY_TYPES}
        multipleFiles={true}
        onUpload={uploadData}
        onDelete={deleteDataFile}
        language={language}
        t={t}
      >
        <IndicatorForm
          groups={groups}
          indicators={indicators}
          onAdd={addIndicator}
          onDelete={deleteIndicator}
          language={language}
          t={t}
        />
      </DataSection>

      {/* Data Overview Matrix */}
      <DataOverviewMatrix
        groups={groups}
        dataFiles={dataFiles}
        indicators={indicators}
        language={language}
        t={t}
      />
    </div>
  )
}
