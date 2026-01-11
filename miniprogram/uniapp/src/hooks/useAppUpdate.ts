/**
 * 应用更新检查 Hook
 * 支持小程序、H5、App 等多平台的自动更新检查
 */

import { ref } from 'vue'

interface UpdateCheckOptions {
  /** 是否自动检查更新，默认 true */
  autoCheck?: boolean
  /** 更新提示标题，默认 "更新提示" */
  title?: string
  /** 更新准备就绪后的提示文案，默认 "新版本已经准备好，是否重启应用？" */
  readyContent?: string
  /** 更新失败的提示文案，默认 "新版本下载失败，请检查网络后重试" */
  failContent?: string
  /** 更新失败是否显示取消按钮，默认 false */
  showCancelOnFail?: boolean
}

interface UpdateResult {
  /** 是否有更新 */
  hasUpdate: boolean
  /** 更新状态：idle（空闲）、checking（检查中）、downloading（下载中）、ready（准备就绪）、failed（失败） */
  status: 'idle' | 'checking' | 'downloading' | 'ready' | 'failed'
  /** 错误信息 */
  error?: string
}

/**
 * 使用应用更新检查
 * @returns 更新检查相关的状态和方法
 */
export function useAppUpdate(options: UpdateCheckOptions = {}) {
  const {
    autoCheck = true,
    title = '更新提示',
    readyContent = '新版本已经准备好，是否重启应用？',
    failContent = '新版本下载失败，请检查网络后重试',
    showCancelOnFail = false,
  } = options

  // 更新状态
  const updateStatus = ref<UpdateResult['status']>('idle')
  const hasUpdate = ref(false)
  const updateError = ref<string>('')

  /**
   * 小程序更新检查
   */
  const checkMiniProgramUpdate = (): Promise<UpdateResult> => {
    return new Promise((resolve) => {
      if (!uni.canIUse('getUpdateManager')) {
        console.log('当前环境不支持更新管理器')
        resolve({ hasUpdate: false, status: 'idle' })
        return
      }

      const updateManager = uni.getUpdateManager()

      // 检测新版本
      updateManager.onCheckForUpdate((res) => {
        hasUpdate.value = res.hasUpdate
        updateStatus.value = res.hasUpdate ? 'downloading' : 'idle'
        console.log('检查更新结果:', res.hasUpdate ? '有新版本' : '暂无新版本')
      })

      // 下载新版本
      updateManager.onUpdateReady(() => {
        updateStatus.value = 'ready'

        uni.showModal({
          title,
          content: readyContent,
          success: (res) => {
            if (res.confirm) {
              // 调用 applyUpdate 应用新版本并重启
              updateManager.applyUpdate()
            } else {
              updateStatus.value = 'idle'
            }
          },
        })
      })

      // 新版本下载失败
      updateManager.onUpdateFailed(() => {
        updateStatus.value = 'failed'
        updateError.value = failContent

        uni.showModal({
          title: '更新失败',
          content: failContent,
          showCancel: showCancelOnFail,
        })

        resolve({ hasUpdate: false, status: 'failed', error: failContent })
      })

      // 立即检查更新
      updateManager.onCheckForUpdate((res) => {
        resolve({
          hasUpdate: res.hasUpdate,
          status: res.hasUpdate ? 'downloading' : 'idle',
        })
      })
    })
  }

  /**
   * H5 更新检查
   */
  const checkH5Update = (): Promise<UpdateResult> => {
    return new Promise((resolve) => {
      console.log('H5 环境，应用自动刷新机制')

      // 检查是否有新版本（通过 API 或其他方式）
      // TODO: 实现实际的版本检查逻辑
      // 示例：请求接口检查版本号
      // uni.request({
      //   url: '/api/version/check',
      //   success: (res) => {
      //     if (res.data.hasUpdate) {
      //       // 提示用户刷新
      //       uni.showModal({
      //         title,
      //         content: '发现新版本，是否刷新页面？',
      //         success: (modalRes) => {
      //           if (modalRes.confirm) {
      //             location.reload()
      //           }
      //         }
      //       })
      //     }
      //   }
      // })

      resolve({ hasUpdate: false, status: 'idle' })
    })
  }

  /**
   * App 更新检查
   */
  const checkAppUpdate = (): Promise<UpdateResult> => {
    return new Promise((resolve) => {
      console.log('App 环境，应用热更新机制')

      // TODO: 实现实际的热更新逻辑
      // 可以使用 uni-app 的热更新插件
      // 示例：
      // uni.request({
      //   url: 'https://example.com/version.json',
      //   success: (res) => {
      //     if (res.data.version > currentVersion) {
      //       // 下载更新包
      //       // 提示用户安装
      //     }
      //   }
      // })

      resolve({ hasUpdate: false, status: 'idle' })
    })
  }

  /**
   * 检查更新
   * 根据当前平台自动选择对应的更新检查方法
   */
  const checkUpdate = async (): Promise<UpdateResult> => {
    updateStatus.value = 'checking'
    updateError.value = ''

    try {
      // #ifdef MP
      return await checkMiniProgramUpdate()
      // #endif

      // #ifdef H5
      return await checkH5Update()
      // #endif

      // #ifdef APP-PLUS
      return await checkAppUpdate()
      // #endif

      // 默认情况
      return { hasUpdate: false, status: 'idle' }
    } catch (error) {
      updateStatus.value = 'failed'
      updateError.value = String(error)
      console.error('更新检查失败:', error)
      return { hasUpdate: false, status: 'failed', error: String(error) }
    }
  }

  /**
   * 手动触发更新检查
   */
  const manualCheckUpdate = async () => {
    uni.showLoading({
      title: '检查更新中...',
    })

    const result = await checkUpdate()

    uni.hideLoading()

    if (!result.hasUpdate && result.status === 'idle') {
      uni.showToast({
        title: '已是最新版本',
        icon: 'success',
      })
    }

    return result
  }

  // 自动检查更新
  if (autoCheck) {
    // 使用 setTimeout 延迟执行，避免阻塞应用启动
    setTimeout(() => {
      checkUpdate()
    }, 1000)
  }

  return {
    // 状态
    updateStatus,
    hasUpdate,
    updateError,

    // 方法
    checkUpdate,
    manualCheckUpdate,

    // 平台特定方法（可选导出）
    checkMiniProgramUpdate,
    checkH5Update,
    checkAppUpdate,
  }
}
