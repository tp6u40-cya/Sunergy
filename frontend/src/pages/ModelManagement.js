// src/pages/ModelManagement.js
import React, { useState } from 'react';
import Navbar from '../components/Navbar';

export default function ModelManagement({ onNavigateToDashboard, onNavigateToTrain, onNavigateToPredict, onNavigateToSites, onNavigateToModelMgmt, onLogout, activePage }) {
  // 模擬已訓練的模型數據
  const [models, setModels] = useState([
    { id: 'M-001', name: 'LSTM_V1_嘉義案場', type: 'LSTM', date: '2025/12/20', accuracy: '96.5%', status: '已部署' },
    { id: 'M-002', name: 'XGB_V2_台南案場', type: 'XGBoost', date: '2025/12/15', accuracy: '94.2%', status: '閒置中' },
    { id: 'M-003', name: 'RF_V1_屏東案場', type: 'RandomForest', date: '2025/11/30', accuracy: '91.8%', status: '閒置中' },
  ]);

  const navProps = { onNavigateToDashboard, onNavigateToTrain, onNavigateToPredict, onNavigateToSites, onNavigateToModelMgmt, onLogout };

  const handleDelete = (id) => {
    if (window.confirm(`確定要刪除模型 ${id} 嗎？`)) {
      setModels(models.filter(m => m.id !== id));
    }
  };

  return (
    <div className="min-h-screen w-full bg-background-dark text-white flex flex-col font-sans">
      {/* 修正：activePage 必須與 App.js 的 'model-mgmt' 一致 */}
      <Navbar activePage="model-mgmt" {...navProps} />

      <main className="flex-1 w-full max-w-7xl mx-auto p-6 py-10">
        <div className="flex flex-col md:flex-row md:items-end justify-between mb-10 border-b border-white/10 pb-6">
          <div>
            <h1 className="text-3xl font-bold tracking-tight">模型管理中心</h1>
            <p className="text-white/40 text-sm mt-1">管理與追蹤所有已訓練完成的 AI 預測模型</p>
          </div>
          <div className="mt-4 md:mt-0 flex gap-4">
             <div className="bg-white/5 border border-white/10 px-4 py-2 rounded-lg">
                <p className="text-[10px] text-white/40 uppercase font-bold">目前模型總數</p>
                <p className="text-xl font-black text-primary">{models.length}</p>
             </div>
          </div>
        </div>

        <div className="grid grid-cols-1 gap-4">
          {models.length > 0 ? models.map((model) => (
            <div key={model.id} className="bg-white/[0.02] border border-white/10 rounded-2xl p-6 flex flex-col md:flex-row md:items-center justify-between hover:bg-white/[0.04] transition-all group">
              <div className="flex items-center gap-6">
                {/* 修正：Material Symbols 名稱應為小寫 psychology 或 memory */}
                <div className="size-14 rounded-xl bg-primary/10 text-primary flex items-center justify-center group-hover:bg-primary group-hover:text-background-dark transition-colors">
                  <span className="material-symbols-outlined !text-3xl">psychology</span>
                </div>
                <div>
                  <div className="flex items-center gap-3">
                    <h3 className="text-lg font-bold text-white group-hover:text-primary transition-colors">{model.name}</h3>
                    <span className={`px-2.5 py-0.5 rounded-full text-[10px] font-black tracking-wider uppercase ${model.status === '已部署' ? 'bg-green-500/20 text-green-400' : 'bg-white/10 text-white/40'}`}>
                      {model.status}
                    </span>
                  </div>
                  <p className="text-xs text-white/40 mt-1.5 font-mono">ID: {model.id} | 算法: {model.type} | 訓練日期: {model.date}</p>
                </div>
              </div>

              <div className="flex items-center justify-between md:justify-end gap-10 mt-6 md:mt-0">
                <div className="text-right">
                  <p className="text-[10px] text-white/40 uppercase font-bold tracking-widest mb-1">訓練準確度</p>
                  <p className="text-2xl font-black text-primary italic">{model.accuracy}</p>
                </div>
                <div className="flex gap-2 border-l border-white/10 pl-6">
                  <button title="查看詳情" className="p-2.5 rounded-xl bg-white/5 hover:bg-white/10 text-white/50 hover:text-white transition-all">
                    <span className="material-symbols-outlined">visibility</span>
                  </button>
                  <button 
                    onClick={() => handleDelete(model.id)} 
                    title="刪除模型"
                    className="p-2.5 rounded-xl bg-red-500/5 hover:bg-red-500/20 text-white/30 hover:text-red-400 transition-all"
                  >
                    <span className="material-symbols-outlined">delete</span>
                  </button>
                </div>
              </div>
            </div>
          )) : (
            <div className="py-20 text-center border-2 border-dashed border-white/5 rounded-3xl">
               <p className="text-white/20 text-lg italic">目前尚無已訓練的模型</p>
            </div>
          )}
        </div>
      </main>

      <footer className="p-8 text-center text-white/10 text-[10px] font-bold uppercase tracking-[0.4em]">
        © 2025 SUNERGY ANALYTICS CENTER
      </footer>
    </div>
  );
}