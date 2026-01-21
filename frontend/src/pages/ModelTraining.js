// src/pages/ModelTraining.js
import React, { useState } from 'react';
import Navbar from '../components/Navbar';

// --- 輔助組件：警示燈 ---
const StatusLight = ({ wmape }) => {
  const value = parseFloat(wmape);
  let status = { color: 'bg-green-500', label: '正常', shadow: 'shadow-[0_0_10px_rgba(34,197,94,0.8)]' };
  if (value > 0.15) status = { color: 'bg-red-500', label: '異常', shadow: 'shadow-[0_0_10px_rgba(239,68,68,0.8)]' };
  else if (value > 0.05) status = { color: 'bg-yellow-500', label: '需留意', shadow: 'shadow-[0_0_10px_rgba(234,179,8,0.8)]' };

  return (
    <div className="flex items-center gap-2">
      <div className={`size-3 rounded-full ${status.color} ${status.shadow}`}></div>
      <span className="text-xs text-white/80">{status.label}</span>
    </div>
  );
};

// --- 輔助組件：日照 vs 發電量圖表 ---
// const SolarProductionChart = ({ results, activeLines }) => {
//   const modelColors = { LSTM: '#60a5fa', XGBoost: '#34d399', RandomForest: '#f2cc0d', SVR: '#a78bfa' };
//   return (
//     <div className="w-full h-80 bg-black/40 rounded-xl p-6 border border-white/10 relative mt-4">
//       <div className="absolute left-2 top-1/2 -translate-y-1/2 -rotate-90 text-[10px] text-white/40 tracking-widest">發電量 (kWh)</div>
//       <svg viewBox="0 0 400 150" className="w-full h-full" preserveAspectRatio="none">
//         <line x1="20" y1="140" x2="400" y2="140" stroke="white" strokeOpacity="0.2" />
//         <line x1="20" y1="10" x2="20" y2="140" stroke="white" strokeOpacity="0.2" />
//         {Object.entries(results).map(([id, res]) => activeLines[id] && (
//           <path
//             key={id}
//             d={`M 20 ${130 - (Math.random()*10)} Q 100 ${110 - (Math.random()*30)}, 200 ${70 - (Math.random()*20)}, 400 ${10 + (Math.random()*20)}`}
//             fill="none"
//             stroke={modelColors[id] || '#fff'}
//             strokeWidth="2"
//             className="transition-all duration-500"
//           />
//         ))}
//       </svg>
//       <div className="text-center text-[10px] text-white/40 tracking-widest mt-2">日照量 (W/m²)</div>
//     </div>
//   );
// };

// --- 輔助組件：可拉動雙點區間調整器 ---
const IntervalSlider = ({ label, min, max, start, end, onStartChange, onEndChange }) => {
  const startPct = ((start - min) / (max - min)) * 100;
  const endPct = ((end - min) / (max - min)) * 100;
  const handleStartMove = (v) => onStartChange(Math.min(v, end));
  const handleEndMove = (v) => onEndChange(Math.max(v, start));

  return (
    <div className="mb-8">
      <div className="flex justify-between items-center mb-3">
        <label className="text-xs font-bold text-primary">{label}</label>
      </div>
      <div className="relative h-6 w-full flex items-center mb-4">
        <div className="absolute h-1 w-full bg-white/10 rounded-full"></div>
        <div className="absolute h-1 bg-primary z-10" style={{ left: `${startPct}%`, right: `${100 - endPct}%` }}></div>
        <input type="range" min={min} max={max} value={end} onChange={(e) => handleEndMove(Number(e.target.value))} className="absolute w-full h-full appearance-none bg-transparent pointer-events-none z-30 accent-white [&::-webkit-slider-thumb]:pointer-events-auto [&::-webkit-slider-thumb]:size-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-primary" />
        <input type="range" min={min} max={max} value={start} onChange={(e) => handleStartMove(Number(e.target.value))} className="absolute w-full h-full appearance-none bg-transparent pointer-events-none z-20 accent-white [&::-webkit-slider-thumb]:pointer-events-auto [&::-webkit-slider-thumb]:size-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-primary" />
      </div>
      <div className="grid grid-cols-2 gap-4">
        <div>
          <p className="text-[9px] text-white/40 mb-1">起始設定</p>
          <input type="number" value={start} onChange={(e) => onStartChange(Number(e.target.value))} className="w-full bg-white/5 border border-white/10 rounded p-2 text-xs text-center focus:border-primary outline-none" />
        </div>
        <div>
          <p className="text-[9px] text-white/40 mb-1">結束設定 (MAX: {max})</p>
          <input type="number" value={end} onChange={(e) => onEndChange(Number(e.target.value))} className="w-full bg-white/5 border border-white/10 rounded p-2 text-xs text-center focus:border-primary outline-none" />
        </div>
      </div>
    </div>
  );
};

export default function ModelTraining({ onBack, onNext, onNavigateToPredict, onLogout, onNavigateToSites }) {
  const [splitRatio, setSplitRatio] = useState(80);
  const [selectedModels, setSelectedModels] = useState(['LSTM']);
  const [paramIntervals, setParamIntervals] = useState({
    LSTM_epochs_s: 50, LSTM_epochs_e: 200,
    XGB_trees_s: 100, XGB_trees_e: 500,
    SVR_c_s: 1, SVR_c_e: 50,
    RF_trees_s: 50, RF_trees_e: 300
  });
  const [activeChartLines, setActiveChartLines] = useState({ LSTM: true, XGBoost: true, SVR: true, RandomForest: true });
  const [isTraining, setIsTraining] = useState(false);
  const [isTrained, setIsTrained] = useState(false);
  const [trainingResults, setTrainingResults] = useState({});

  const toggleModel = (id) => setSelectedModels(prev => prev.includes(id) ? prev.filter(m => m !== id) : [...prev, id]);

  const handleStartTraining = () => {
    if (selectedModels.length === 0) return alert('請選擇模型');
    setIsTraining(true);
    setTimeout(() => {
      const mock = selectedModels.reduce((acc, id) => {
        acc[id] = { id, r2: (0.91 + Math.random() * 0.07).toFixed(3), rmse: (0.1 + Math.random() * 1.4).toFixed(2), mae: (0.1 + Math.random() * 0.9).toFixed(2), wmape: (Math.random() * 0.18).toFixed(4) };
        return acc;
      }, {});
      setTrainingResults(mock);
      setIsTraining(false);
      setIsTrained(true);
    }, 2000);
  };

  return (
    <div className="min-h-screen w-full bg-background-dark text-white flex flex-col font-sans">
      <Navbar activePage="predict" onNavigateToDashboard={onNavigateToPredict} onNavigateToPredict={onNavigateToPredict} onLogout={onLogout} onNavigateToSites={onNavigateToSites} />

      {/* [新增] Sticky Header 步驟指示器 */}
      <div className="w-full border-b border-white/10 bg-white/[.02] px-6 py-3 sticky top-[64px] sm:top-[65px] z-40 backdrop-blur-md">
        <div className="mx-auto flex max-w-5xl items-center justify-between">
          <button onClick={onBack} className="flex items-center gap-1 text-sm text-white/50 hover:text-white transition-colors">
            <span className="material-symbols-outlined !text-lg">arrow_back</span>
            返回上一步
          </button>
          
          <div className="text-sm font-medium">
            <span className="text-white/40">1. 上傳資料</span>
            <span className="mx-2 text-white/30">/</span>
            <span className="text-white/40">2. 清理資料</span>
            <span className="mx-2 text-white/30">/</span>
            <span className="text-white/40 ">3. 調整單位</span>
            <span className="mx-2 text-white/30">/</span>
            <span className="text-primary font-bold">4. 模型訓練與優化</span>
            
            
          </div>
        </div>
      </div>

      <main className="flex-1 w-full max-w-7xl mx-auto p-6 py-10 grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* 左側：設定流程 */}
        <div className="lg:col-span-4 flex flex-col gap-8">
          {/* Step 1: 分割資料 */}
          <section className="bg-white/[0.02] p-5 rounded-2xl border border-white/5">
            <h2 className="text-sm font-bold text-primary mb-4 flex items-center gap-2">
              <span className="size-5 rounded-full bg-primary text-background-dark flex items-center justify-center text-[10px]">1</span> 分割資料
            </h2>
            <div className="flex justify-between text-[10px] text-white/50 mb-2"><span>訓練集 {splitRatio}%</span><span>測試集 {100-splitRatio}%</span></div>
            <input type="range" min="50" max="95" step="5" value={splitRatio} onChange={(e) => setSplitRatio(e.target.value)} className="w-full h-1 bg-white/10 rounded-lg appearance-none cursor-pointer accent-primary" />
          </section>

          {/* Step 2: 選擇模型 */}
          <section className="bg-white/[0.02] p-5 rounded-2xl border border-white/5">
            <h2 className="text-sm font-bold text-primary mb-4 flex items-center gap-2">
              <span className="size-5 rounded-full bg-primary text-background-dark flex items-center justify-center text-[10px]">2</span> 選擇模型
            </h2>
            <div className="grid grid-cols-2 gap-2">
              {['LSTM', 'XGBoost', 'SVR', 'RandomForest'].map(m => (
                <button key={m} onClick={() => toggleModel(m)} className={`p-2 rounded-lg border text-[10px] font-bold transition-all ${selectedModels.includes(m) ? 'border-primary bg-primary/10 text-primary' : 'border-white/10 bg-white/5 text-white/40'}`}>{m}</button>
              ))}
            </div>
          </section>

          {/* Step 3: 調整參數 */}
          <section className="bg-white/[0.02] p-5 rounded-2xl border border-white/5">
            <h2 className="text-sm font-bold text-primary mb-4 flex items-center gap-2">
              <span className="size-5 rounded-full bg-primary text-background-dark flex items-center justify-center text-[10px]">3</span> 調整參數
            </h2>
            <div className="flex flex-col gap-4">
              {selectedModels.map(id => (
                <div key={id} className="p-4 bg-black/20 rounded-xl border border-white/5">
                  <p className="text-[10px] font-bold text-white/60 mb-6 uppercase tracking-tighter border-b border-white/5 pb-1">{id} 模型區間設定</p>
                  {id === 'LSTM' && <IntervalSlider label="Epochs (訓練次數)" min={1} max={1000} start={paramIntervals.LSTM_epochs_s} end={paramIntervals.LSTM_epochs_e} onStartChange={(v)=>setParamIntervals({...paramIntervals, LSTM_epochs_s: v})} onEndChange={(v)=>setParamIntervals({...paramIntervals, LSTM_epochs_e: v})} />}
                  {id === 'XGBoost' && <IntervalSlider label="n_estimators (樹木數量)" min={10} max={2000} start={paramIntervals.XGB_trees_s} end={paramIntervals.XGB_trees_e} onStartChange={(v)=>setParamIntervals({...paramIntervals, XGB_trees_s: v})} onEndChange={(v)=>setParamIntervals({...paramIntervals, XGB_trees_e: v})} />}
                  {id === 'SVR' && <IntervalSlider label="C (懲罰參數區間)" min={0.1} max={100} start={paramIntervals.SVR_c_s} end={paramIntervals.SVR_c_e} onStartChange={(v)=>setParamIntervals({...paramIntervals, SVR_c_s: v})} onEndChange={(v)=>setParamIntervals({...paramIntervals, SVR_c_e: v})} />}
                  {id === 'RandomForest' && <IntervalSlider label="n_estimators (森林規模)" min={10} max={1000} start={paramIntervals.RF_trees_s} end={paramIntervals.RF_trees_e} onStartChange={(v)=>setParamIntervals({...paramIntervals, RF_trees_s: v})} onEndChange={(v)=>setParamIntervals({...paramIntervals, RF_trees_e: v})} />}
                </div>
              ))}
              {selectedModels.length === 0 && <p className="text-center text-[10px] text-white/20 italic py-4">請先在步驟 2 選擇模型</p>}
            </div>
          </section>

          <button onClick={handleStartTraining} disabled={isTraining || selectedModels.length === 0} className="w-full py-4 bg-primary text-background-dark rounded-xl font-black text-sm transition-all hover:scale-[1.02] active:scale-95 disabled:opacity-30">
            {isTraining ? '模型訓練中...' : '開始執行訓練'}
          </button>
        </div>

        {/* 右側：結果顯示 */}
        <div className="lg:col-span-8 flex flex-col gap-6">
          <div className={`flex-1 rounded-2xl border border-white/10 bg-white/[0.01] p-6 relative overflow-hidden ${!isTrained && 'flex items-center justify-center border-dashed'}`}>
            {isTrained ? (
              <div className="w-full animate-fade-in">
                <div className="flex justify-between items-center mb-6">
                   <h2 className="text-xl font-bold">訓練結果看板</h2>
                   {/* <div className="flex gap-2">
                      {Object.keys(trainingResults).map(id => (
                        <label key={id} className="flex items-center gap-1 cursor-pointer">
                          <input type="checkbox" checked={activeChartLines[id]} onChange={()=>setActiveChartLines({...activeChartLines, [id]: !activeChartLines[id]})} className="size-3 accent-primary" />
                          <span className="text-[10px] text-white/60">{id}</span>
                        </label>
                      ))}
                   </div> */}
                </div>

                {/* <SolarProductionChart results={trainingResults} activeLines={activeChartLines} /> */}

                <div className="mt-8 overflow-hidden rounded-xl border border-white/10">
                  <table className="w-full text-xs text-left">
                    <thead className="bg-white/5 text-white/40 uppercase">
                      <tr><th className="px-4 py-3">模型</th><th className="px-4 py-3">狀態</th><th className="px-4 py-3 text-right">R² Score</th><th className="px-4 py-3 text-right">RMSE (kW)</th><th className="px-4 py-3 text-right">MAE (kW)</th><th className="px-4 py-3 text-right">WMAPE</th></tr>
                    </thead>
                    <tbody className="divide-y divide-white/5 font-mono text-white/80">
                      {Object.values(trainingResults).map(res => (
                        <tr key={res.id} className="hover:bg-white/5 transition-colors">
                          <td className="px-4 py-4 font-bold text-primary font-sans">{res.id}</td>
                          <td className="px-4 py-4"><StatusLight wmape={res.wmape} /></td>
                          <td className="px-4 py-4 text-right text-green-400">{res.r2}</td>
                          <td className="px-4 py-4 text-right">{res.rmse}</td>
                          <td className="px-4 py-4 text-right">{res.mae}</td>
                          <td className="px-4 py-4 text-right text-yellow-500/80">{res.wmape}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ) : (
              <div className="text-center opacity-20">
                <span className="material-symbols-outlined !text-6xl mb-2">query_stats</span>
                <h1 className="lg font-bold text-white mb-2">等待訓練</h1>
                <p className="text-white/100 text-sm max-w-xs mx-auto">請在左側選擇您欲比較的所有模型，然後點擊「開始訓練」。</p>
              </div>
            )}
          </div>
        </div>
      </main>

      <div className="p-6 border-t border-white/10 bg-background-dark/90 flex justify-end gap-4">
        <button onClick={onBack} className="px-6 py-2 text-white/40 hover:text-white transition-colors text-sm">取消</button>
        <button onClick={onNext} disabled={!isTrained} className={`px-10 py-2 rounded-lg font-bold text-sm transition-all ${isTrained ? 'bg-primary text-background-dark hover:shadow-[0_0_15px_rgba(242,204,13,0.4)]' : 'bg-white/10 text-white/20 cursor-not-allowed'}`}>開始進行預測</button>
      </div>
    </div>
  );
}