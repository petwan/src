/**
 * Store 类型定义文件
 * 定义所有 store 模块的状态类型和操作类型
 */

import { IUserInfo } from '../typings'

/**
 * Pinia Store 的持久化配置
 */
export interface PersistConfig {
  /** 是否启用持久化 */
  enabled?: boolean
  /** 存储的 key */
  key?: string
  /** 要持久化的状态字段 */
  paths?: string[]
}

/**
 * 用户 Store 状态类型
 */
export interface UserState {
  /** 用户信息 */
  userInfo: IUserInfo | null
  /** 登录令牌 */
  token: string
  /** 是否已登录 */
  isLogin: boolean
}

/**
 * 应用 Store 状态类型
 */
export interface AppState {
  /** 主题模式 */
  theme: 'light' | 'dark' | 'auto'
  /** 语言 */
  locale: string
  /** 网络状态 */
  networkStatus: {
    /** 是否在线 */
    isConnected: boolean
    /** 网络类型 */
    networkType: string
  }
  /** 设备信息 */
  deviceInfo: {
    /** 平台 */
    platform: string
    /** 系统版本 */
    system: string
    /** 机型 */
    model: string
  }
}

/**
 * 通用 Store State 类型
 */
export interface BaseState {
  /** 加载状态 */
  loading: boolean
  /** 错误信息 */
  error: string | null
}

/**
 * 分页状态类型
 */
export interface PaginationState {
  /** 当前页码 */
  currentPage: number
  /** 每页条数 */
  pageSize: number
  /** 总条数 */
  total: number
  /** 是否还有更多 */
  hasMore: boolean
}

/**
 * 通用分页 Store 状态
 */
export interface PaginationListState<T> extends BaseState {
  /** 数据列表 */
  list: T[]
  /** 分页信息 */
  pagination: PaginationState
}

/**
 * Store 的 Actions 参数类型
 */
export interface LoginParams {
  /** 用户名 */
  username: string
  /** 密码 */
  password: string
  /** 验证码 */
  captcha?: string
}

/**
 * 更新用户信息参数
 */
export interface UpdateUserInfoParams {
  /** 昵称 */
  nickname?: string
  /** 头像 */
  avatar?: string
  /** 手机号 */
  telephone?: number
}

/**
 * 隐私 Store 状态类型
 */
export interface PrivacyState {
  /** 是否同意隐私政策 */
  hasAgreed: boolean
  /** 隐私政策版本 */
  version: string
  /** 同意时间 */
  agreedAt: number | null
  /** 是否已显示隐私弹窗 */
  hasShownDialog: boolean
}

/**
 * 隐私政策信息
 */
export interface PrivacyPolicyInfo {
  /** 版本号 */
  version: string
  /** 隐私政策 URL */
  url: string
  /** 服务条款 URL */
  termsUrl: string
  /** 隐私政策更新时间 */
  updateTime: number
}
