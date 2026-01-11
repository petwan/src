/**
 * 隐私 Store 模块
 * 用于审计和记录用户隐私政策同意状态
 */

import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { PrivacyState, PrivacyPolicyInfo } from '../types'
import { getPlatform } from '../../utils/device'

/**
 * 隐私 Store ID
 */
export const PRIVACY_STORE_ID = 'privacy'

/**
 * 默认隐私政策配置
 */
const DEFAULT_PRIVACY_CONFIG: PrivacyPolicyInfo = {
  version: '1.0.0',
  url: 'https://example.com/privacy',
  termsUrl: 'https://example.com/terms',
  updateTime: Date.now(),
}

/**
 * 隐私 Store
 */
export const usePrivacyStore = defineStore(
  PRIVACY_STORE_ID,
  () => {
    // ==================== State ====================
    const hasAgreed = ref<boolean>(false)
    const version = ref<string>(DEFAULT_PRIVACY_CONFIG.version)
    const agreedAt = ref<number | null>(null)
    const hasShownDialog = ref<boolean>(false)

    // ==================== Getters ====================
    /**
     * 是否需要显示隐私弹窗
     */
    const shouldShowDialog = computed(() => {
      return !hasAgreed.value && !hasShownDialog.value
    })

    /**
     * 隐私协议是否有效(版本未过期)
     */
    const isAgreementValid = computed(() => {
      if (!hasAgreed.value || !agreedAt.value) return false
      // 检查当前同意的版本是否为最新版本
      return version.value === DEFAULT_PRIVACY_CONFIG.version
    })

    // ==================== Actions ====================
    /**
     * 同意隐私政策
     */
    const agreePrivacyPolicy = () => {
      hasAgreed.value = true
      version.value = DEFAULT_PRIVACY_CONFIG.version
      agreedAt.value = Date.now()
      hasShownDialog.value = true

      // 记录审计日志
      logPrivacyAudit('agree')
    }

    /**
     * 拒绝隐私政策
     */
    const declinePrivacyPolicy = () => {
      hasAgreed.value = false
      hasShownDialog.value = true

      // 记录审计日志
      logPrivacyAudit('decline')

      // 拒绝后退出应用
      uni.reLaunch({
        url: '/pages/index/index',
      })
    }

    /**
     * 标记已显示隐私弹窗
     */
    const markDialogShown = () => {
      hasShownDialog.value = true
    }

    /**
     * 重置隐私状态(用于测试或隐私政策更新)
     */
    const resetPrivacyState = () => {
      hasAgreed.value = false
      version.value = DEFAULT_PRIVACY_CONFIG.version
      agreedAt.value = null
      hasShownDialog.value = false

      logPrivacyAudit('reset')
    }

    /**
     * 检查隐私政策是否需要重新同意
     * 当隐私政策版本更新时调用
     */
    const checkPrivacyUpdate = () => {
      if (!isAgreementValid.value) {
        // 版本不一致,需要重新同意
        hasAgreed.value = false
        hasShownDialog.value = false

        logPrivacyAudit('version_update')
      }
    }

    /**
     * 记录隐私审计日志
     */
    const logPrivacyAudit = (action: 'agree' | 'decline' | 'reset' | 'version_update') => {
      const platform = getPlatform()

      const auditLog = {
        action,
        timestamp: Date.now(),
        version: version.value,
        hasAgreed: hasAgreed.value,
        agreedAt: agreedAt.value,
        platform,
        // 可以根据需要添加更多审计信息
      }

      console.log('[隐私审计]', auditLog)

      // TODO: 将审计日志发送到服务器
      // uni.request({
      //   url: '/api/audit/log',
      //   method: 'POST',
      //   data: auditLog,
      // })
    }

    /**
     * 初始化隐私状态
     */
    const initPrivacy = () => {
      // 检查隐私政策是否更新
      checkPrivacyUpdate()
    }

    /**
     * 获取隐私政策信息
     */
    const getPrivacyPolicyInfo = (): PrivacyPolicyInfo => {
      return DEFAULT_PRIVACY_CONFIG
    }

    return {
      // State
      hasAgreed,
      version,
      agreedAt,
      hasShownDialog,
      // Getters
      shouldShowDialog,
      isAgreementValid,
      // Actions
      agreePrivacyPolicy,
      declinePrivacyPolicy,
      markDialogShown,
      resetPrivacyState,
      checkPrivacyUpdate,
      initPrivacy,
      getPrivacyPolicyInfo,
    }
  },
  {
    persist: {
      key: 'privacy-store',
      paths: ['hasAgreed', 'version', 'agreedAt', 'hasShownDialog'],
    },
  },
)

// 导出类型供外部使用
export type PrivacyStore = ReturnType<typeof usePrivacyStore>
