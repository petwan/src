/**
 * 设备信息工具函数
 * 提供统一的设备信息获取接口，适配不同平台和 API 版本
 */

/**
 * 获取平台信息
 * @returns 平台信息
 */
export function getPlatform(): string {
  // #ifdef MP-WEIXIN
  try {
    // @ts-ignore - 新 API 可能在旧版本中不存在
    const appBaseInfo = wx.getAppBaseInfo()
    // @ts-ignore - 新 API 可能在旧版本中不存在
    const deviceBaseInfo = wx.getDeviceInfo()
    return deviceBaseInfo?.platform || appBaseInfo?.platform || ''
  } catch (error) {
    console.warn('获取平台信息失败，使用兼容方式:', error)
    return uni.getSystemInfoSync().platform
  }
  // #endif

  // #ifndef MP-WEIXIN
  return uni.getSystemInfoSync().platform
  // #endif
}

/**
 * 获取系统信息
 * @returns 系统信息
 */
export function getSystem(): string {
  // #ifdef MP-WEIXIN
  try {
    // @ts-ignore - 新 API 可能在旧版本中不存在
    const appBaseInfo = wx.getAppBaseInfo()
    // @ts-ignore - 新 API 可能在旧版本中不存在
    const deviceBaseInfo = wx.getDeviceInfo()
    return deviceBaseInfo?.system || appBaseInfo?.system || ''
  } catch (error) {
    console.warn('获取系统信息失败，使用兼容方式:', error)
    return uni.getSystemInfoSync().system
  }
  // #endif

  // #ifndef MP-WEIXIN
  return uni.getSystemInfoSync().system
  // #endif
}

/**
 * 获取设备型号
 * @returns 设备型号
 */
export function getModel(): string {
  // #ifdef MP-WEIXIN
  try {
    // @ts-ignore - 新 API 可能在旧版本中不存在
    const deviceBaseInfo = wx.getDeviceInfo()
    return deviceBaseInfo?.model || ''
  } catch (error) {
    console.warn('获取设备型号失败，使用兼容方式:', error)
    return uni.getSystemInfoSync().model
  }
  // #endif

  // #ifndef MP-WEIXIN
  return uni.getSystemInfoSync().model
  // #endif
}

/**
 * 获取屏幕宽度
 * @returns 屏幕宽度
 */
export function getWindowWidth(): number {
  // #ifdef MP-WEIXIN
  try {
    // @ts-ignore - 新 API 可能在旧版本中不存在
    const windowInfo = wx.getWindowInfo()
    return windowInfo?.windowWidth || 0
  } catch (error) {
    console.warn('获取屏幕宽度失败，使用兼容方式:', error)
    return uni.getSystemInfoSync().windowWidth
  }
  // #endif

  // #ifndef MP-WEIXIN
  return uni.getSystemInfoSync().windowWidth
  // #endif
}

/**
 * 获取屏幕高度
 * @returns 屏幕高度
 */
export function getWindowHeight(): number {
  // #ifdef MP-WEIXIN
  try {
    // @ts-ignore - 新 API 可能在旧版本中不存在
    const windowInfo = wx.getWindowInfo()
    return windowInfo?.windowHeight || 0
  } catch (error) {
    console.warn('获取屏幕高度失败，使用兼容方式:', error)
    return uni.getSystemInfoSync().windowHeight
  }
  // #endif

  // #ifndef MP-WEIXIN
  return uni.getSystemInfoSync().windowHeight
  // #endif
}

/**
 * 获取状态栏高度
 * @returns 状态栏高度
 */
export function getStatusBarHeight(): number {
  // #ifdef MP-WEIXIN
  try {
    // @ts-ignore - 新 API 可能在旧版本中不存在
    const windowInfo = wx.getWindowInfo()
    return windowInfo?.statusBarHeight || 0
  } catch (error) {
    console.warn('获取状态栏高度失败，使用兼容方式:', error)
    return uni.getSystemInfoSync().statusBarHeight
  }
  // #endif

  // #ifndef MP-WEIXIN
  return uni.getSystemInfoSync().statusBarHeight
  // #endif
}

/**
 * 获取系统设置
 * @returns 系统设置信息
 */
export function getSystemSetting() {
  // #ifdef MP-WEIXIN
  try {
    // @ts-ignore - 新 API 可能在旧版本中不存在
    return wx.getSystemSetting()
  } catch (error) {
    console.warn('获取系统设置失败:', error)
    return {}
  }
  // #endif

  // #ifndef MP-WEIXIN
  return {}
  // #endif
}

/**
 * 获取应用授权设置
 * @returns 授权设置信息
 */
export function getAppAuthorizeSetting() {
  // #ifdef MP-WEIXIN
  try {
    // @ts-ignore - 新 API 可能在旧版本中不存在
    return wx.getAppAuthorizeSetting()
  } catch (error) {
    console.warn('获取应用授权设置失败:', error)
    return {}
  }
  // #endif

  // #ifndef MP-WEIXIN
  return {}
  // #endif
}

/**
 * 获取完整的设备信息对象
 * @returns 设备信息对象
 */
export function getDeviceInfo() {
  return {
    platform: getPlatform(),
    system: getSystem(),
    model: getModel(),
    windowWidth: getWindowWidth(),
    windowHeight: getWindowHeight(),
    statusBarHeight: getStatusBarHeight(),
  }
}

