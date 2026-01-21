// src/pages/PredictSolar.js
import React, { useState } from 'react';
import Navbar from '../components/Navbar';

// --- 重新設計：專業診斷指標組件 (含警示燈) ---
const DiagnosticMetrics = ({ errorValue, totalPower, avgIrradiance }) => {
  const value = parseFloat(errorValue);
  let status = { 
    color: 'text-green-400', 
    bgColor: 'bg-green-500', 
    label: '發電正常', 
    shadow: 'shadow-[0_0_15px_rgba(34,197,94,0.4)]',
    desc: '預測與實際高度吻合'
  };
  
  if (value > 15) {
    status = { 
      color: 'text-red-400', 
      bgColor: 'bg-red-500', 
      label: '發電異常', 
      shadow: 'shadow-[0_0_15px_rgba(239,68,68,0.4)]',
      desc: '偏差過大，請檢查設備'
    };
  } else if (value > 5) {
    status = { 
      color: 'text-yellow-400', 
      bgColor: 'bg-yellow-500', 
      label: '需留意', 
      shadow: 'shadow-[0_0_15px_rgba(234,179,8,0.4)]',
      desc: '環境干擾或輕微積塵'
    };
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
      {/* 總發電量卡片 */}
      <div className="bg-white/5 border border-white/10 p-5 rounded-2xl flex flex-col justify-center">
        <p className="text-[10px] text-white/40 uppercase font-bold mb-2 tracking-widest">預估發電總量</p>
        <div className="flex items-baseline gap-2">
          <span className="text-3xl font-black font-mono text-white">{totalPower}</span>
          <span className="text-xs text-primary font-bold">kWh</span>
        </div>
      </div>

      {/* 平均日照卡片 */}
      <div className="bg-white/5 border border-white/10 p-5 rounded-2xl flex flex-col justify-center">
        <p className="text-[10px] text-white/40 uppercase font-bold mb-2 tracking-widest">平均日照強度</p>
        <div className="flex items-baseline gap-2">
          <span className="text-3xl font-black font-mono text-white">{avgIrradiance}</span>
          <span className="text-xs text-primary font-bold">W/m²</span>
        </div>
      </div>

      {/* 警示燈診斷卡片 - 現在與前兩者完全對齊 */}
      <div className="bg-white/5 border border-white/10 p-5 rounded-2xl flex items-center gap-4">
        <div className={`relative flex items-center justify-center`}>
          <div className={`size-10 rounded-full ${status.bgColor} ${status.shadow} animate-pulse`}></div>
          <div className="absolute size-14 rounded-full border border-white/5 animate-ping opacity-20"></div>
        </div>
        <div className="flex flex-col">
          <p className="text-[10px] text-white/40 uppercase font-bold tracking-widest mb-0.5">系統診斷狀態</p>
          <span className={`text-lg font-black ${status.color}`}>{status.label}</span>
          <span className="text-[10px] text-white/30 italic">{status.desc}</span>
        </div>
      </div>
    </div>
  );
};

// --- 對比圖表 (優化座標軸與圖例) ---
const ComparisonChart = () => (
  <div className="w-full flex-1 min-h-[350px] bg-black/40 rounded-2xl p-8 border border-white/10 relative mt-4 flex flex-col">
    <div className="flex justify-between items-center mb-6">
      <h3 className="text-xs font-bold text-white/50 uppercase tracking-widest flex items-center gap-2">
        <span className="material-symbols-outlined !text-sm">stacked_line_chart</span>
        發電相關性分析圖
      </h3>
      <div className="flex gap-4 text-[10px] font-bold">
        <span className="flex items-center gap-2 text-white/60">
          <div className="w-4 h-0.5 bg-blue-400 opacity-40 border-b border-dashed"></div> 實際值
        </span>
        <span className="flex items-center gap-2 text-primary">
          <div className="w-4 h-1 bg-primary rounded-full"></div> 預測值
        </span>
      </div>
    </div>
    
    <div className="flex-1 relative">
      {/* Y 軸標籤 */}
      <div className="absolute -left-10 top-1/2 -translate-y-1/2 -rotate-90 text-[9px] text-primary/60 font-bold tracking-widest whitespace-nowrap">
        日照強度 (W/m²)
      </div>
      
      <svg viewBox="0 0 400 150" className="w-full h-full" preserveAspectRatio="none">
        <line x1="25" y1="10" x2="25" y2="135" stroke="white" strokeOpacity="0.1" />
        <line x1="25" y1="135" x2="400" y2="135" stroke="white" strokeOpacity="0.1" />
        {/* 模擬動態路徑 */}
        <path d="M 25 135 L 80 110 L 160 85 L 260 55 L 380 20" fill="none" stroke="#60a5fa" strokeWidth="1.5" strokeOpacity="0.3" strokeDasharray="4 2" />
        <path d="M 25 130 L 85 105 L 165 80 L 265 50 L 385 15" fill="none" stroke="#f2cc0d" strokeWidth="2.5" strokeLinecap="round" />
      </svg>
      
      {/* X 軸標籤 */}
      <div className="absolute -bottom-2 right-0 text-[9px] text-primary/60 font-bold tracking-widest">
        預估發電量 (kWh)
      </div>
    </div>
  </div>
);

export default function PredictSolar({ onBack, onNavigateToDashboard, onLogout, onNavigateToSites, onNavigateToTrain, onNavigateToPredict, onNavigateToModelMgmt }) {
  const [file, setFile] = useState(null);
  const [selectedTrainedModel, setSelectedTrainedModel] = useState('');
  const [isPredicting, setIsPredicting] = useState(false);
  const [hasResult, setHasResult] = useState(false);

  const trainedHistory = [
    { id: 'm1', date: '2025/12/20', model: 'LSTM', accuracy: '96.5%' },
    { id: 'm2', date: '2025/12/15', model: 'XGBoost', accuracy: '94.2%' },
    { id: 'm3', date: '2025/11/30', model: 'RandomForest', accuracy: '91.8%' },
  ];

  const handlePredict = () => {
    if (!file || !selectedTrainedModel) return alert("請上傳資料並選擇訓練紀錄。");
    setIsPredicting(true);
    setTimeout(() => {
      setIsPredicting(false);
      setHasResult(true);
    }, 2000);
  };

  const navProps = { onNavigateToDashboard, onNavigateToTrain, onNavigateToPredict, onNavigateToSites, onNavigateToModelMgmt, onLogout };

  return (
    <div className="min-h-screen w-full bg-background-dark text-white flex flex-col font-sans">
      <Navbar activePage="predict-solar" {...navProps} />

      <main className="flex-1 w-full max-w-7xl mx-auto p-6 py-10 grid grid-cols-1 lg:grid-cols-12 gap-8 items-stretch">
        
        {/* 左側：配置區 */}
        <div className="lg:col-span-4 flex flex-col">
          <section className="bg-white/[0.02] p-8 rounded-[2rem] border border-white/10 flex-1 flex flex-col shadow-2xl">
            <h2 className="text-lg font-bold text-white mb-8 flex items-center gap-3 italic">
              <div className="size-8 rounded-lg bg-primary/20 text-primary flex items-center justify-center">
                <span className="material-symbols-outlined !text-xl">settings_applications</span>
              </div>
              預測配置中心
            </h2>
            
            <div className="flex-1 space-y-8">
              <div>
                <label className="text-[11px] text-white/30 mb-3 block font-bold uppercase tracking-widest">1. 上傳預測原始資料</label>
                <div 
                  className="group border-2 border-dashed border-white/10 rounded-2xl p-8 text-center hover:bg-white/[0.03] hover:border-primary/50 transition-all cursor-pointer"
                  onClick={() => document.getElementById('fileInput').click()}
                >
                  <input type="file" id="fileInput" hidden onChange={(e) => setFile(e.target.files[0])} />
                  <div className="size-12 rounded-full bg-white/5 mx-auto mb-4 flex items-center justify-center group-hover:bg-primary/10 transition-colors">
                    <span className="material-symbols-outlined !text-2xl text-white/30 group-hover:text-primary">upload_file</span>
                  </div>
                  <p className="text-xs font-bold text-white/50 group-hover:text-white transition-colors">{file ? file.name : "CSV / XLSX 檔案"}</p>
                </div>
              </div>

              <div>
                <label className="text-[11px] text-white/30 mb-3 block font-bold uppercase tracking-widest">2. 選擇歷史訓練模型</label>
                <select 
                  value={selectedTrainedModel} 
                  onChange={(e) => setSelectedTrainedModel(e.target.value)}
                  className="w-full bg-black/40 border border-white/10 rounded-xl px-4 py-4 text-sm text-white focus:ring-2 focus:ring-primary/50 outline-none appearance-none cursor-pointer"
                >
                  <option value="">請選擇版本...</option>
                  {trainedHistory.map(h => (
                    <option key={h.id} value={h.id}>
                      {h.date} - {h.model} (Acc: {h.accuracy})
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <button 
              onClick={handlePredict}
              disabled={isPredicting || !file || !selectedTrainedModel}
              className="w-full bg-primary text-background-dark py-5 rounded-2xl font-black text-sm hover:scale-[1.02] active:scale-95 transition-all shadow-[0_10px_30px_rgba(242,204,13,0.2)] mt-8"
            >
              {isPredicting ? "運算執行中..." : "開始執行預測分析"}
            </button>
          </section>
        </div>

        {/* 右側：結果看板 */}
        <div className="lg:col-span-8 flex flex-col">
          <div className={`flex-1 w-full rounded-[2rem] border border-white/10 bg-white/[0.01] p-10 flex flex-col relative transition-all shadow-2xl ${!hasResult && 'items-center justify-center border-dashed opacity-40'}`}>
            
            {hasResult ? (
              <div className="w-full h-full flex flex-col animate-fade-in">
                <div className="flex justify-between items-start mb-8">
                   <h2 className="text-2xl font-black italic tracking-tighter">分析預測結果<br/><span className="text-primary text-sm font-bold uppercase tracking-[0.3em]">發電預測即時診斷</span></h2>
                   <div className="text-right">
                      <p className="text-[10px] text-white/20 uppercase font-bold tracking-widest">Model Version</p>
                      <p className="text-xs font-mono text-white/60 italic">{trainedHistory.find(h=>h.id===selectedTrainedModel)?.date} Fixed</p>
                   </div>
                </div>

                {/* 指標區域 - 三張卡片現在水平對齊且視覺統一 */}
                <DiagnosticMetrics totalPower="1,428.5" avgIrradiance="682.4" errorValue="4.2" />

                <ComparisonChart />

                <p className="mt-8 text-[11px] text-white/20 leading-relaxed italic text-center">
                  * 本預測結果基於選定之訓練權重。若診斷異常請重新檢查 EAC 資料單位或重新進行模型訓練。
                </p>
              </div>
            ) : (
              <div className="text-center space-y-4">
                <div className="size-20 rounded-full bg-white/5 mx-auto flex items-center justify-center">
                  <span className="material-symbols-outlined !text-4xl text-white/10">query_stats</span>
                </div>
                <p className="text-sm font-bold text-white/20 tracking-widest uppercase">等待配置與預測執行</p>
              </div>
            )}
          </div>
        </div>
      </main>

      {/* 底部導航 */}
      <div className="p-8 border-t border-white/10 bg-background-dark/95 flex justify-end gap-6 backdrop-blur-xl">
        <button onClick={onBack} className="text-xs font-bold text-white/30 hover:text-white transition-colors">回模型訓練</button>
        <button onClick={onNavigateToDashboard} className="px-10 py-3 rounded-xl bg-white/5 border border-white/10 text-white font-bold text-xs hover:bg-white/10 hover:border-white/20 transition-all">返回首頁看板</button>
      </div>
    </div>
  );
}