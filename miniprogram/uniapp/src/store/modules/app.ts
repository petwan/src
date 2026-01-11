/**
 * 应用 Store 模块
 * 管理应用全局状态,如主题、语言、设备信息等
 */

import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { AppState } from '../types'
import { getDeviceInfo } from '../../utils/device'

/**
 * 应用 Store ID
 */
export const APP_STORE_ID = 'app'

/**
 * 应用 Store
 */
export const useAppStore = defineStore(
  APP_STORE_ID,
  () => {
    // ==================== State ====================
    const theme = ref<AppState['theme']>('light')
    const locale = ref<string>('zh-CN')
    const networkStatus = ref<AppState['networkStatus']>({
      isConnected: true,
      networkType: 'wifi',
    })
    const deviceInfo = ref<AppState['deviceInfo']>({
      platform: '',
      system: '',
      model: '',
    })

    // ==================== Getters ====================
    const isDark = computed(() => theme.value === 'dark')
    const isOnline = computed(() => networkStatus.value.isConnected)

    // ==================== Actions ====================
    /**
     * 设置主题
     */
    const setTheme = (newTheme: AppState['theme']) => {
      theme.value = newTheme
      // 可以在这里添加主题切换的副作用逻辑
      // 例如更新 uni-app 的导航栏样式等
    }

    /**
     * 切换主题
     */
    const toggleTheme = () => {
      theme.value = theme.value === 'light' ? 'dark' : 'light'
    }

    /**
     * 设置语言
     */
    const setLocale = (newLocale: string) => {
      locale.value = newLocale
      uni.setLocale(newLocale)
    }

    /**
     * 更新网络状态
     */
    const updateNetworkStatus = (status: Partial<AppState['networkStatus']>) => {
      networkStatus.value = { ...networkStatus.value, ...status }
    }

    /**
     * 设置设备信息
     */
    const setDeviceInfo = (info: Partial<AppState['deviceInfo']>) => {
      deviceInfo.value = { ...deviceInfo.value, ...info }
    }

    /**
     * 初始化设备信息
     */
    const initDeviceInfo = () => {
      const deviceData = getDeviceInfo()
      deviceInfo.value = {
        platform: deviceData.platform,
        system: deviceData.system,
        model: deviceData.model,
      }
    }

    /**
     * 初始化网络监听
     */
    const initNetworkListener = () => {
      uni.getNetworkType({
        success: (res) => {
          updateNetworkStatus({
            networkType: res.networkType,
          })
        },
      })

      uni.onNetworkStatusChange((res) => {
        updateNetworkStatus({
          isConnected: res.isConnected,
          networkType: res.networkType,
        })
      })
    }

    /**
     * 初始化应用状态
     */
    const initApp = () => {
      initDeviceInfo()
      initNetworkListener()
    }

    return {
      // State
      theme,
      locale,
      networkStatus,
      deviceInfo,
      // Getters
      isDark,
      isOnline,
      // Actions
      setTheme,
      toggleTheme,
      setLocale,
      updateNetworkStatus,
      setDeviceInfo,
      initDeviceInfo,
      initNetworkListener,
      initApp,
    }
  },
  {
    persist: {
      key: 'app-store',
      paths: ['theme', 'locale'],
    },
  },
)

// 导出类型供外部使用
export type AppStore = ReturnType<typeof useAppStore>
