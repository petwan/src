/**
 * Store 使用示例
 * 展示如何在实际业务中使用各个 Store 模块
 */

import { useUserStore } from './user'
import { useAppStore } from './app'
import { usePrivacyStore } from './privacy'
import { watch } from 'vue'
import type { AppState } from '../types'

/**
 * 示例 1: 在 Vue 组件中使用 Store
 * ================================
 */

/**
 * 在页面组件中的使用示例
 */
export const exampleComponentUsage = () => {
  // 1. 导入 Store
  const userStore = useUserStore()
  const appStore = useAppStore()
  const privacyStore = usePrivacyStore()

  // 2. 访问 State
  console.log('用户信息:', userStore.userInfo)
  console.log('当前主题:', appStore.theme)
  console.log('是否已同意隐私:', privacyStore.hasAgreed)

  // 3. 访问 Getters
  console.log('是否已登录:', userStore.isLogin)
  console.log('是否为暗色模式:', appStore.isDark)
  console.log('是否在线:', appStore.isOnline)
  console.log('是否需要显示隐私弹窗:', privacyStore.shouldShowDialog)

  // 4. 调用 Actions
  // 登录
  userStore.login({
    username: 'user123',
    password: 'password123',
  })

  // 切换主题
  appStore.toggleTheme()

  // 设置语言
  appStore.setLocale('en-US')

  // 同意隐私政策
  privacyStore.agreePrivacyPolicy()

  // 退出登录
  userStore.logout()
}

/**
 * 示例 2: 在页面生命周期中使用 Store
 * ====================================
 */
export const exampleLifecycleUsage = {
  onLoad() {
    const appStore = useAppStore()
    const privacyStore = usePrivacyStore()

    // 初始化应用状态
    appStore.initApp()

    // 检查隐私政策
    privacyStore.initPrivacy()

    // 如果需要显示隐私弹窗
    if (privacyStore.shouldShowDialog) {
      // 显示隐私弹窗
      uni.showModal({
        title: '隐私政策',
        content: '请阅读并同意隐私政策',
        success: (res) => {
          if (res.confirm) {
            privacyStore.agreePrivacyPolicy()
          } else {
            privacyStore.declinePrivacyPolicy()
          }
        },
      })
    }
  },

  onShow() {
    const appStore = useAppStore()

    // 检查网络状态
    if (!appStore.isOnline) {
      uni.showToast({
        title: '网络未连接',
        icon: 'none',
      })
    }
  },
}

/**
 * 示例 3: 在 API 请求中使用 Store
 * =================================
 */
export const exampleApiUsage = async () => {
  const userStore = useUserStore()

  // 在请求头中添加 token
  const token = userStore.token

  const response = await uni.request({
    url: '/api/user/profile',
    method: 'GET',
    header: {
      Authorization: `Bearer ${token}`,
    },
  })

  return response
}

/**
 * 示例 4: Store 间的交互
 * ========================
 */
export const exampleStoreInteraction = () => {
  const userStore = useUserStore()
  const privacyStore = usePrivacyStore()

  // 登录前检查隐私政策
  if (!privacyStore.hasAgreed) {
    console.warn('用户未同意隐私政策,无法登录')
    return
  }

  // 登录成功后设置用户信息
  userStore.login({
    username: 'user123',
    password: 'password123',
  })
}

/**
 * 示例 5: 使用 TypeScript 类型
 * ============================
 */
export const exampleTypeUsage = () => {
  // 可以定义参数类型
  const updateAppState = (_state: Partial<AppState>) => {
    const appStore = useAppStore()
    // 更新应用状态
  }

  // 返回类型推断
  const getUserInfo = () => {
    const userStore = useUserStore()
    return userStore.userInfo // 类型为 IUserInfo | null
  }
}

/**
 * 示例 6: 监听 Store 变化
 * ========================
 */
export const exampleWatchStore = () => {
  const appStore = useAppStore()

  // 监听主题变化
  watch(
    () => appStore.theme,
    (newTheme: AppState['theme']) => {
      console.log('主题已切换:', newTheme)
      // 执行主题切换后的操作,比如更新页面样式
    },
  )

  // 监听语言变化
  watch(
    () => appStore.locale,
    (newLocale: string) => {
      console.log('语言已切换:', newLocale)
      // 重新加载页面或更新文案
    },
  )
}

/**
 * 示例 7: 在组合式函数中使用 Store
 * ================================
 */
export const useUserProfile = () => {
  const userStore = useUserStore()

  // 获取用户信息
  const getUserProfile = async () => {
    if (!userStore.isLogin) {
      throw new Error('用户未登录')
    }

    const response = await userStore.fetchUserInfo()
    userStore.setUserInfo(response[1].data)

    return response[1].data
  }

  // 更新用户信息
  const updateUserProfile = async (params: { nickname?: string; avatar?: string }) => {
    if (!userStore.isLogin) {
      throw new Error('用户未登录')
    }

    const response = await userStore.updateUserInfo(params)
    return response
  }

  return {
    getUserProfile,
    updateUserProfile,
  }
}

/**
 * 示例 8: 在工具函数中使用 Store
 * ==============================
 */
export const exampleUtilsUsage = () => {
  /**
   * 检查隐私协议的装饰器函数
   */
  const withPrivacyCheck = <T extends (...args: any[]) => any>(fn: T): T => {
    return (async (...args: any[]) => {
      const privacyStore = usePrivacyStore()

      if (!privacyStore.hasAgreed) {
        uni.showToast({
          title: '请先同意隐私政策',
          icon: 'none',
        })
        return
      }

      return fn(...args)
    }) as T
  }

  // 使用示例
  const uploadFile = withPrivacyCheck(async (file: File) => {
    // 上传文件逻辑
    console.log('上传文件:', file)
  })

  return { uploadFile }
}

/**
 * 示例 9: 重置所有 Store 状态
 * ===========================
 */
export const exampleResetAllStores = () => {
  const userStore = useUserStore()
  const privacyStore = usePrivacyStore()

  // 清空用户信息
  userStore.logout()

  // 重置隐私状态
  privacyStore.resetPrivacyState()
}

/**
 * 示例 10: 在 uni-app 页面中使用
 * =============================
 */
export const examplePageUsage = {
  data() {
    return {
      // 可以将 Store 的状态映射到 data
      userInfo: null as any,
      username: '',
      password: '',
    }
  },

  onLoad(this: any) {
    const userStore = useUserStore()
    const appStore = useAppStore()
    const privacyStore = usePrivacyStore()

    // 同步数据到 data
    this.userInfo = userStore.userInfo

    // 初始化应用
    appStore.initApp()

    // 检查隐私政策
    privacyStore.initPrivacy()
  },

  methods: {
    handleLogin(this: any) {
      const userStore = useUserStore()

      userStore
        .login({
          username: this.username,
          password: this.password,
        })
        .then(() => {
          uni.showToast({
            title: '登录成功',
            icon: 'success',
          })
        })
        .catch(() => {
          uni.showToast({
            title: '登录失败',
            icon: 'none',
          })
        })
    },

    handleLogout() {
      const userStore = useUserStore()
      userStore.logout()
    },

    handleThemeToggle() {
      const appStore = useAppStore()
      appStore.toggleTheme()
    },
  },
}

/**
 * 导出所有示例
 */
export default {
  componentUsage: exampleComponentUsage,
  lifecycleUsage: exampleLifecycleUsage,
  apiUsage: exampleApiUsage,
  storeInteraction: exampleStoreInteraction,
  typeUsage: exampleTypeUsage,
  watchStore: exampleWatchStore,
  useUserProfile,
  utilsUsage: exampleUtilsUsage,
  resetAllStores: exampleResetAllStores,
  pageUsage: examplePageUsage,
}

