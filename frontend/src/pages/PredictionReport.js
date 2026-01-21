// src/pages/PredictionReport.js
import React, { useState } from 'react';
import Navbar from '../components/Navbar';

// --- 模擬複合式圖表 (SVG) ---
const PredictionChart = () => (
  <div className="relative w-full h-80 mt-6">
    <svg className="w-full h-full" viewBox="0 0 800 300" preserveAspectRatio="none">
      {/* 格線 (Y軸) */}
      {[0, 1, 2, 3, 4].map((i) => (<line key={i} x1="40" y1={260 - i * 60} x2="760" y2={260 - i * 60} stroke="rgba(255,255,255,0.1)" strokeWidth="1" />))}

      {/* Y軸文字 */}
      <text x="10" y="20" fill="rgba(255,255,255,0.4)" fontSize="10">450</text><text x="10" y="80" fill="rgba(255,255,255,0.4)" fontSize="10">350</text><text x="10" y="140" fill="rgba(255,255,255,0.4)" fontSize="10">250</text><text x="10" y="200" fill="rgba(255,255,255,0.4)" fontSize="10">150</text><text x="10" y="260" fill="rgba(255,255,255,0.4)" fontSize="10">50</text>

      {/* 長條圖 (預測發電量 - 黃色) */}
      {[
        0, 0, 0, 0, 5, 20, 60, 120, 180, 230, 250, 230, 180, 120, 60, 20, 5, 0, 0, 0, 0, 0, 0, 0
      ].map((val, index) => (
        <rect 
          key={index}
          x={50 + index * 30} 
          y={260 - val} 
          width="12" 
          height={val} 
          fill="transparent" 
          stroke="#f2cc0d" 
          strokeWidth="1"
          rx="2"
        />
      ))}
      {/* 填滿的長條圖 */}
      {[
        0, 0, 0, 0, 5, 20, 60, 120, 180, 230, 250, 230, 180, 120, 60, 20, 5, 0, 0, 0, 0, 0, 0, 0
      ].map((val, index) => (
        <rect 
          key={`fill-${index}`}
          x={50 + index * 30} 
          y={260 - val} 
          width="12" 
          height={val} 
          fill="#f2cc0d" 
          opacity="0.2"
          rx="2"
        />
      ))}

      {/* 折線圖 1 (日照量 - 粉紅色) */}
      <path 
        d="M 50 260 C 150 260, 200 260, 380 20 S 560 260, 740 260" 
        fill="none" 
        stroke="#f472b6" 
        strokeWidth="2"
      />

      {/* 折線圖 2 (氣溫 - 藍色) */}
      <path 
        d="M 50 240 L 150 235 L 250 220 L 380 180 L 500 190 L 650 230 L 740 245" 
        fill="none" 
        stroke="#38bdf8" 
        strokeWidth="2"
        strokeDasharray="4"
      />

      {/* X軸時間標籤 */}
      {['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '23:00'].map((time, i) => (
        <text key={i} x={50 + i * 110} y="280" fill="rgba(255,255,255,0.4)" fontSize="10" textAnchor="middle">{time}</text>
      ))}
    </svg>

    {/* 圖例 */}
    <div className="absolute top-0 right-4 flex gap-4">
      <div className="flex items-center gap-2">
        <div className="w-3 h-3 bg-[#38bdf8] rounded-sm"></div>
        <span className="text-xs text-white/60">氣溫 (°C)</span>
      </div>
      <div className="flex items-center gap-2">
        <div className="w-3 h-3 bg-[#f472b6] rounded-sm"></div>
        <span className="text-xs text-white/60">日照量 (W/m²)</span>
      </div>
      <div className="flex items-center gap-2">
        <div className="w-3 h-3 border border-[#f2cc0d] bg-[#f2cc0d]/20 rounded-sm"></div>
        <span className="text-xs text-white/60">預測發電量 (kWh)</span>
      </div>
    </div>
  </div>
);

export default function PredictionReport({ onBack, onNavigateToDashboard, onNavigateToPredict, onNavigateToSites, onLogout }) {
  
  return (
    <div className="min-h-screen w-full bg-background-dark text-white flex flex-col">
      <Navbar 
        activePage="predict"
        onNavigateToDashboard={onNavigateToDashboard}
        onNavigateToPredict={onNavigateToPredict}
        onNavigateToSites={onNavigateToSites}
        onLogout={onLogout}
      />

      {/* Sticky Header (步驟進度) */}
      <div className="w-full border-b border-white/10 bg-white/[.02] px-6 py-3 sticky top-[64px] sm:top-[65px] z-40 backdrop-blur-md">
        <div className="mx-auto flex max-w-7xl items-center justify-between">
          <button onClick={onNavigateToDashboard} className="flex items-center gap-1 text-sm text-white/50 hover:text-white transition-colors">
             <span className="material-symbols-outlined !text-lg">arrow_back</span>
             返回首頁
          </button>
          
          <div className="text-sm font-medium hidden sm:block">
            <span className="text-white/40">1. 上傳資料</span><span className="mx-2 text-white/30">&gt;</span>
            <span className="text-white/40">2. 清理資料</span><span className="mx-2 text-white/30">&gt;</span>
            <span className="text-white/40">3. 調整單位</span><span className="mx-2 text-white/30">&gt;</span>
            <span className="text-white/40">4. 模型訓練與資料分割</span><span className="mx-2 text-white/30">&gt;</span>
            <span className="text-primary font-bold">5. 輸出結果</span>
          </div>
        </div>
      </div>

      {/* 3. 主要內容區 */}
      <main className="flex-1 w-full max-w-7xl mx-auto p-6 py-8 flex flex-col gap-6">
        
        {/* 標題與重新預測 */}
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-bold text-white">2025/11/15 發電量預測報告</h1>

        </div>

        {/* --- 頂部 4 張 KPI 卡片 --- */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Card 1 */}
          <div className="rounded-xl border border-white/10 bg-white/[.03] p-5">
            <p className="text-xs text-white/50 mb-1">未來 24 小時總發電量</p>
            <div className="flex items-end gap-2 mb-2">
              <span className="text-3xl font-bold text-white">4,850</span>
              <span className="text-sm text-white/60 mb-1">kWh</span>
            </div>
            <div className="flex items-center gap-1 text-xs text-green-400">
              <span className="material-symbols-outlined !text-sm">trending_up</span>
              較昨日增加 5.2%
            </div>
          </div>
          {/* Card 2 */}
          <div className="rounded-xl border border-white/10 bg-white/[.03] p-5">
            <p className="text-xs text-white/50 mb-1">預估收益</p>
            <div className="flex items-end gap-2 mb-2">
              <span className="text-3xl font-bold text-white">$21,825</span>
              <span className="text-sm text-white/60 mb-1">TWD</span>
            </div>
            <div className="text-xs text-white/40">費率: $4.5 / kWh</div>
          </div>
          {/* Card 3 */}
          <div className="rounded-xl border border-white/10 bg-white/[.03] p-5">
            <p className="text-xs text-white/50 mb-1">最高發電峰值</p>
            <div className="flex items-end gap-2 mb-2">
              <span className="text-3xl font-bold text-white">450</span>
              <span className="text-sm text-white/60 mb-1">kW</span>
            </div>
            <div className="text-xs text-white/40">預計出現在 13:00</div>
          </div>
          {/* Card 4 */}
          <div className="rounded-xl border border-white/10 bg-white/[.03] p-5">
            <p className="text-xs text-white/50 mb-1">減碳效益</p>
            <div className="flex items-end gap-2 mb-2">
              <span className="text-3xl font-bold text-white">2,468</span>
              <span className="text-sm text-white/60 mb-1">kgCO₂e</span>
            </div>
            <div className="text-xs text-white/40">相當於 205 棵樹木一年的吸碳量</div>
          </div>
        </div>

        {/* --- 中間：視覺化預測圖表 (已修改為垂直堆疊容器) --- */}
        <div className="rounded-xl border border-white/10 bg-white/[.03] p-6 flex flex-col gap-6">
            <h3 className="text-lg font-bold text-white mb-2">模型比較與預測曲線</h3>
            
            {/* 1. 預測曲線與實際值 (主要圖表) */}
            <div className="border-b border-white/10 pb-4">
                <h4 className="text-sm font-bold text-white/80 mb-2">預測曲線與實際值 (多模型對比)</h4>
                <PredictionChart /> 
            </div>

            {/* 2. 誤差殘差分析 (次級圖表 - 垂直堆疊) */}
            <div className="pb-4">
                <h4 className="text-sm font-bold text-white/80 mb-2">誤差殘差分佈</h4>
                {/* Placeholder for Residuals Chart */}
                <div className="w-full h-48 bg-black/20 rounded-lg flex items-center justify-center text-white/50 border border-dashed border-white/10">
                    <span className="material-symbols-outlined text-4xl text-white/40 mr-2">analytics</span>
                    [殘差分佈圖表 Placeholder - 需在前端解析多模型數據]
                </div>
            </div>
        </div>


        {/* --- 底部區塊：左右分欄 --- */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          
          {/* 左下：匯出與應用 */}
          <div className="flex flex-col gap-4 rounded-xl border border-white/10 bg-white/[.03] p-6">
            <h3 className="text-lg font-bold text-white">匯出與應用</h3>
            
            {/* 表格預覽 */}
            <div className="overflow-hidden rounded-lg border border-white/10">
              <table className="w-full text-sm text-left text-white/70">
                <thead className="bg-white/5 text-xs uppercase text-white/50">
                  <tr>
                    <th className="px-4 py-3 font-medium">時間</th>
                    <th className="px-4 py-3 font-medium text-right">發電量 (kWh)</th>
                    <th className="px-4 py-3 font-medium text-right">氣溫 (°C)</th>
                    <th className="px-4 py-3 font-medium text-right">日照量 (W/m²)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5">
                  {[
                    { time: '00:00', power: 0, temp: 22, sun: 0 },
                    { time: '01:00', power: 0, temp: 21.5, sun: 0 },
                    { time: '02:00', power: 0, temp: 21, sun: 0 },
                  ].map((row, idx) => (
                    <tr key={idx} className="hover:bg-white/[.02]">
                      <td className="px-4 py-3 font-mono text-white/90">{row.time}</td>
                      <td className="px-4 py-3 text-right font-mono">{row.power}</td>
                      <td className="px-4 py-3 text-right font-mono">{row.temp}</td>
                      <td className="px-4 py-3 text-right font-mono">{row.sun}</td>
                    </tr>
                  ))}
                  <tr>
                    <td colSpan="4" className="px-4 py-2 text-center text-xs text-white/30">...</td>
                  </tr>
                </tbody>
              </table>
            </div>

            {/* 匯出按鈕群 */}
            <div className="flex gap-3 mt-auto pt-2">
              <button className="flex-1 flex items-center justify-center gap-2 rounded-lg border border-white/10 py-2.5 text-sm font-medium text-white hover:bg-white/10 transition-colors">
                <span className="material-symbols-outlined !text-lg">download</span>
                下載 CSV
              </button>
              <button className="flex-1 flex items-center justify-center gap-2 rounded-lg border border-white/10 py-2.5 text-sm font-medium text-white hover:bg-white/10 transition-colors">
                <span className="material-symbols-outlined !text-lg">picture_as_pdf</span>
                下載 PDF
              </button>
              <button className="flex-1 flex items-center justify-center gap-2 rounded-lg bg-primary py-2.5 text-sm font-bold text-background-dark hover:scale-105 transition-transform">
                <span className="material-symbols-outlined !text-lg">api</span>
                透過 API 傳送
              </button>
            </div>
          </div>

          {/* 右下：異常預警 + 回儀表板 */}
          <div className="flex flex-col gap-4">
            <h3 className="text-lg font-bold text-white">異常預警與健康診斷</h3>
            
            <div className="rounded-lg border border-white/10 bg-green-500/5 p-4 flex gap-4 items-start">
              <div className="flex size-10 items-center justify-center rounded-full bg-green-500/20 text-green-500 flex-shrink-0">
                <span className="material-symbols-outlined">check_circle</span>
              </div>
              <div>
                <h4 className="text-base font-bold text-white mb-1">系統運作狀態：正常</h4>
                <p className="text-sm text-white/60">所有感測器與逆變器回報狀態良好。</p>
              </div>
            </div>

            <div className="rounded-lg border border-yellow-500/20 bg-yellow-500/5 p-4 flex gap-4 items-start">
              <div className="flex size-10 items-center justify-center rounded-full bg-yellow-500/20 text-yellow-500 flex-shrink-0">
                <span className="material-symbols-outlined">warning</span>
              </div>
              <div>
                <h4 className="text-base font-bold text-white mb-1">潛在警示：發電量多次為０</h4>
                <p className="text-sm text-white/60 leading-relaxed">
                  #3 號逆變器串列電流略低於預期，建議安排人員檢查問題。
                </p>
              </div>
            </div>

            <div className="mt-auto pt-4 flex justify-end">
              <button 
                onClick={onBack} // 回到模型訓練頁
                className="flex items-center gap-2 rounded-lg border border-white/10 px-4 py-2 text-sm font-medium text-white hover:bg-white/10 transition-colors"
              >
                <span className="material-symbols-outlined !text-lg">refresh</span>
                重新預測
              </button>
              <button 
                onClick={onNavigateToDashboard}
                className="flex items-center justify-center gap-2 rounded-lg bg-primary px-8 py-3 text-base font-bold text-background-dark transition-transform hover:scale-105"
              >
                回到首頁
                
              </button>
            </div>
          </div>

        </div>

      </main>
    </div>
  );
}