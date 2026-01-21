// src/pages/Settings.js
import React, { useState } from 'react';
import Navbar from '../components/Navbar';

export default function Settings({ onNavigateToDashboard, onNavigateToPredict, onNavigateToSites, onLogout, onNavigateToSettings }) {
  const [activeTab, setActiveTab] = useState('account');
  const [userName, setUserName] = useState('User');
  const [userEmail, setUserEmail] = useState('user@example.com');
  const [darkMode, setDarkMode] = useState(true);
  const [notifications, setNotifications] = useState(true);

  return (
    <div className="min-h-screen w-full bg-background-dark text-white flex flex-col">
      <Navbar 
        activePage="settings"
        onNavigateToDashboard={onNavigateToDashboard}
        onNavigateToPredict={onNavigateToPredict}
        onNavigateToSites={onNavigateToSites}
        onLogout={onLogout}
        onNavigateToSettings={onNavigateToSettings}
      />

      <main className="flex-1 w-full max-w-6xl mx-auto p-6 py-10">
        <div className="flex items-center justify-between mb-8">
          <h1 className="text-3xl font-bold text-white">設定</h1>
          <button onClick={onNavigateToDashboard} className="flex items-center gap-1 text-sm text-white/50 hover:text-white transition-colors">
            <span className="material-symbols-outlined !text-lg">close</span>
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          {/* 側邊標籤 */}
          <div className="flex flex-col gap-2">
            <button 
              onClick={() => setActiveTab('account')}
              className={`text-left px-4 py-3 rounded-lg transition-colors ${activeTab === 'account' ? 'bg-primary/20 text-primary font-bold border border-primary/50' : 'text-white/70 hover:text-white hover:bg-white/5'}`}
            >
              帳號設定
            </button>
            <button 
              onClick={() => setActiveTab('privacy')}
              className={`text-left px-4 py-3 rounded-lg transition-colors ${activeTab === 'privacy' ? 'bg-primary/20 text-primary font-bold border border-primary/50' : 'text-white/70 hover:text-white hover:bg-white/5'}`}
            >
              隱私與安全
            </button>
            <button 
              onClick={() => setActiveTab('preferences')}
              className={`text-left px-4 py-3 rounded-lg transition-colors ${activeTab === 'preferences' ? 'bg-primary/20 text-primary font-bold border border-primary/50' : 'text-white/70 hover:text-white hover:bg-white/5'}`}
            >
              偏好設定
            </button>
            <button 
              onClick={() => setActiveTab('notifications')}
              className={`text-left px-4 py-3 rounded-lg transition-colors ${activeTab === 'notifications' ? 'bg-primary/20 text-primary font-bold border border-primary/50' : 'text-white/70 hover:text-white hover:bg-white/5'}`}
            >
              通知設定
            </button>
            <button 
              onClick={() => setActiveTab('about')}
              className={`text-left px-4 py-3 rounded-lg transition-colors ${activeTab === 'about' ? 'bg-primary/20 text-primary font-bold border border-primary/50' : 'text-white/70 hover:text-white hover:bg-white/5'}`}
            >
              關於應用
            </button>
          </div>

          {/* 內容區域 */}
          <div className="md:col-span-3">
            {/* 帳號設定 */}
            {activeTab === 'account' && (
              <div className="rounded-xl border border-white/10 bg-white/[.02] p-8">
                <h2 className="text-2xl font-bold mb-6">帳號設定</h2>
                
                <div className="space-y-6">
                  <div>
                    <label className="block text-sm font-medium text-white/70 mb-2">使用者名稱</label>
                    <input 
                      type="text" 
                      value={userName} 
                      onChange={(e) => setUserName(e.target.value)}
                      className="w-full rounded-lg border border-white/10 bg-black/20 px-4 py-3 text-white placeholder-white/40 focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-white/70 mb-2">電子郵件</label>
                    <input 
                      type="email" 
                      value={userEmail} 
                      onChange={(e) => setUserEmail(e.target.value)}
                      className="w-full rounded-lg border border-white/10 bg-black/20 px-4 py-3 text-white placeholder-white/40 focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
                    />
                  </div>

                  <div className="pt-4">
                    <button className="px-6 py-2.5 rounded-lg bg-primary text-background-dark font-bold hover:scale-105 transition-transform">
                      保存變更
                    </button>
                  </div>
                </div>
              </div>
            )}

            {/* 隱私與安全 */}
            {activeTab === 'privacy' && (
              <div className="rounded-xl border border-white/10 bg-white/[.02] p-8">
                <h2 className="text-2xl font-bold mb-6">隱私與安全</h2>
                
                <div className="space-y-6">
                  <div className="flex items-center justify-between p-4 rounded-lg bg-white/5 border border-white/10">
                    <div>
                      <p className="font-medium text-white">變更密碼</p>
                      <p className="text-sm text-white/50">定期更改您的密碼以保護帳戶安全</p>
                    </div>
                    <button className="px-4 py-2 rounded-lg bg-white/10 text-white/70 hover:bg-white/20 transition-colors">
                      變更
                    </button>
                  </div>

                  <div className="flex items-center justify-between p-4 rounded-lg bg-white/5 border border-white/10">
                    <div>
                      <p className="font-medium text-white">雙因素認證</p>
                      <p className="text-sm text-white/50">加強帳戶安全性</p>
                    </div>
                    <button className="px-4 py-2 rounded-lg bg-white/10 text-white/70 hover:bg-white/20 transition-colors">
                      啟用
                    </button>
                  </div>

                  <div className="flex items-center justify-between p-4 rounded-lg bg-white/5 border border-white/10">
                    <div>
                      <p className="font-medium text-white">活動日誌</p>
                      <p className="text-sm text-white/50">查看您的帳戶活動記錄</p>
                    </div>
                    <button className="px-4 py-2 rounded-lg bg-white/10 text-white/70 hover:bg-white/20 transition-colors">
                      檢視
                    </button>
                  </div>
                </div>
              </div>
            )}

            {/* 偏好設定 */}
            {activeTab === 'preferences' && (
              <div className="rounded-xl border border-white/10 bg-white/[.02] p-8">
                <h2 className="text-2xl font-bold mb-6">偏好設定</h2>
                
                <div className="space-y-6">
                  <div className="flex items-center justify-between p-4 rounded-lg bg-white/5 border border-white/10">
                    <div>
                      <p className="font-medium text-white">深色模式</p>
                      <p className="text-sm text-white/50">使用深色介面</p>
                    </div>
                    <button 
                      onClick={() => setDarkMode(!darkMode)}
                      className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${darkMode ? 'bg-primary' : 'bg-white/20'}`}
                    >
                      <span className={`inline-block h-5 w-5 transform rounded-full bg-white transition-transform ${darkMode ? 'translate-x-5' : 'translate-x-0'}`}></span>
                    </button>
                  </div>

                  <div className="flex items-center justify-between p-4 rounded-lg bg-white/5 border border-white/10">
                    <div>
                      <p className="font-medium text-white">預設語言</p>
                      <p className="text-sm text-white/50">選擇介面語言</p>
                    </div>
                    <select className="rounded-lg border border-white/10 bg-black/20 px-4 py-2 text-white focus:border-primary focus:outline-none">
                      <option>繁體中文</option>
                      <option>English</option>
                    </select>
                  </div>
                </div>
              </div>
            )}

            {/* 通知設定 */}
            {activeTab === 'notifications' && (
              <div className="rounded-xl border border-white/10 bg-white/[.02] p-8">
                <h2 className="text-2xl font-bold mb-6">通知設定</h2>
                
                <div className="space-y-6">
                  <div className="flex items-center justify-between p-4 rounded-lg bg-white/5 border border-white/10">
                    <div>
                      <p className="font-medium text-white">預測完成提醒</p>
                      <p className="text-sm text-white/50">當預測完成時發送通知</p>
                    </div>
                    <button 
                      onClick={() => setNotifications(!notifications)}
                      className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${notifications ? 'bg-primary' : 'bg-white/20'}`}
                    >
                      <span className={`inline-block h-5 w-5 transform rounded-full bg-white transition-transform ${notifications ? 'translate-x-5' : 'translate-x-0'}`}></span>
                    </button>
                  </div>

                  <div className="flex items-center justify-between p-4 rounded-lg bg-white/5 border border-white/10">
                    <div>
                      <p className="font-medium text-white">每日報告摘要</p>
                      <p className="text-sm text-white/50">每天接收系統摘要報告</p>
                    </div>
                    <button className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors bg-primary`}>
                      <span className={`inline-block h-5 w-5 transform rounded-full bg-white transition-transform translate-x-5`}></span>
                    </button>
                  </div>
                </div>
              </div>
            )}

            {/* 關於應用 */}
            {activeTab === 'about' && (
              <div className="rounded-xl border border-white/10 bg-white/[.02] p-8">
                <h2 className="text-2xl font-bold mb-6">關於應用</h2>
                
                <div className="space-y-6">
                  <div className="p-4 rounded-lg bg-white/5 border border-white/10">
                    <p className="text-sm text-white/70 mb-2">應用名稱</p>
                    <p className="text-lg font-medium text-white">日光預 - 太陽光發電量預測平台</p>
                  </div>

                  <div className="p-4 rounded-lg bg-white/5 border border-white/10">
                    <p className="text-sm text-white/70 mb-2">版本</p>
                    <p className="text-lg font-medium text-white">v1.0.0</p>
                  </div>

                  <div className="p-4 rounded-lg bg-white/5 border border-white/10">
                    <p className="text-sm text-white/70 mb-2">說明</p>
                    <p className="text-sm text-white/80">使用機器學習模型預測太陽光發電量，幫助您優化案場管理。</p>
                  </div>

                  <div className="pt-4 space-y-3 flex flex-col">
                    <button className="px-4 py-2 rounded-lg border border-white/10 text-white/70 hover:text-white hover:bg-white/5 transition-colors text-left">
                      📄 使用條款
                    </button>
                    <button className="px-4 py-2 rounded-lg border border-white/10 text-white/70 hover:text-white hover:bg-white/5 transition-colors text-left">
                      🔒 隱私政策
                    </button>
                    <button className="px-4 py-2 rounded-lg border border-white/10 text-white/70 hover:text-white hover:bg-white/5 transition-colors text-left">
                      💬 聯絡我們
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
