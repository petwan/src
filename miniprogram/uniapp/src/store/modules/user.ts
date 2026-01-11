/**
 * 用户 Store 模块
 * 管理用户登录状态、用户信息等
 */

import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { LoginParams, UpdateUserInfoParams } from '../types'
import type { IUserInfo } from '../../typings'

/**
 * 用户 Store ID
 */
export const USER_STORE_ID = 'user'

/**
 * 用户 Store
 */
export const useUserStore = defineStore(
  USER_STORE_ID,
  () => {
    // ==================== State ====================
    const userInfo = ref<IUserInfo | null>(null)
    const token = ref<string>('')
    const isLogin = computed(() => !!token.value && !!userInfo.value)

    // ==================== Actions ====================
    /**
     * 设置用户信息
     */
    const setUserInfo = (info: IUserInfo | null) => {
      userInfo.value = info
    }

    /**
     * 设置令牌
     */
    const setToken = (newToken: string) => {
      token.value = newToken
    }

    /**
     * 登录
     */
    const login = async (params: LoginParams) => {
      try {
        // TODO: 实现登录逻辑
        const response = await uni.request({
          url: '/api/login',
          method: 'POST',
          data: params,
        })
        return response
      } catch (error) {
        console.error('登录失败:', error)
        throw error
      }
    }

    /**
     * 退出登录
     */
    const logout = () => {
      userInfo.value = null
      token.value = ''
      uni.reLaunch({
        url: '/pages/index/index',
      })
    }

    /**
     * 更新用户信息
     */
    const updateUserInfo = async (params: UpdateUserInfoParams) => {
      try {
        // TODO: 实现更新用户信息逻辑
        const response = await uni.request({
          url: '/api/user/update',
          method: 'POST',
          data: params,
        })
        if (userInfo.value) {
          userInfo.value = { ...userInfo.value, ...params }
        }
        return response
      } catch (error) {
        console.error('更新用户信息失败:', error)
        throw error
      }
    }

    /**
     * 获取用户信息
     */
    const fetchUserInfo = async () => {
      try {
        // TODO: 实现获取用户信息逻辑
        const response = await uni.request({
          url: '/api/user/info',
          method: 'GET',
        })
        return response
      } catch (error) {
        console.error('获取用户信息失败:', error)
        throw error
      }
    }

    return {
      // State
      userInfo,
      token,
      isLogin,
      // Actions
      setUserInfo,
      setToken,
      login,
      logout,
      updateUserInfo,
      fetchUserInfo,
    }
  },
  {
    persist: {
      key: 'user-store',
      paths: ['userInfo', 'token'],
    },
  },
)

// 导出类型供外部使用
export type UserStore = ReturnType<typeof useUserStore>
