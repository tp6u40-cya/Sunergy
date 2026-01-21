// src/pages/PublicHome.js
import React from 'react';

export default function PublicHome({ onOpenLogin, onOpenUserGuide }) {
  
  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-background-dark text-white">
      
      {/* 居中容器 */}
      <div className="flex items-center justify-center w-full max-w-2xl px-4"> 
       <div className="flex-grow border-t-4 border-white/40"></div>
        <div className="mx-6 sm:mx-12 flex items-center justify-center relative"> 
          <div className="relative flex items-center justify-center p-4"> 
            <span className="text-7xl sm:text-8xl font-black tracking-widest text-primary z-20 
                       text-stroke text-stroke-2 text-primary [text-shadow:0_0_20px_#f2cc0d,0_0_40px_#f2cc0d80,0_0_60px_#f2cc0d40] 
                       animate-fade-in-up"> 
              日光預
            </span>
            <div className="absolute inset-0 m-auto size-32 sm:size-40 -z-0 flex items-center justify-center"> 
              <svg viewBox="0 0 100 100" className="w-full h-full animate-pulse" style={{ filter: 'drop-shadow(0 0 15px rgba(242, 204, 13, 0.6))' }}>
                <path d="M 50, 5 A 45 45 0 0 0 50 95 Z" fill="#f2cc0d" opacity="0.3" />
                <path d="M 50, 5 A 45 45 0 0 1 50 95" fill="none" stroke="#f2cc0d" strokeWidth="1.5" opacity="0.4" />
              </svg>
            </div>
          </div>
        </div>
        <div className="flex-grow border-t-4 border-white/40"></div>
      </div>

      {/* 按鈕區域 */}
      <div className="mt-20 flex flex-col gap-4 items-center w-full max-w-xs">
        {/* 1. 進入系統 (主按鈕) */}
        <button
          onClick={onOpenLogin}
          className="w-full flex items-center justify-center gap-3 rounded-lg bg-primary px-10 py-4 text-xl font-bold text-background-dark shadow-lg 
                     transition-all duration-300 hover:scale-105 hover:[box-shadow:0_0_20px_#f2cc0d60] 
                     animate-bounce-once" 
        >
          進入系統
          <span className="material-symbols-outlined !text-2xl ml-2 animate-bounce-horizontal">arrow_forward</span> 
        </button>

        {/* 2. 使用者教學 (次要按鈕) - 新增 */}
        <button
          onClick={onOpenUserGuide}
          className="flex items-center justify-center gap-2 rounded-lg border border-white/10 bg-white/5 px-6 py-2.5 text-sm font-medium text-white/70 
                     transition-all duration-300 hover:bg-white/10 hover:text-white hover:border-white/30"
        >
          <span className="material-symbols-outlined !text-lg">school</span>
          使用者教學
        </button>
      </div>

      {/* Footer */}
      <footer className="absolute bottom-0 w-full border-t border-solid border-white/10 p-4 text-center">
        <p className="text-white/50 text-sm">© 2025 日光預. All rights reserved.</p>
        <div className="flex justify-center gap-6 mt-2">
            <a className="text-white/50 hover:text-white/80 text-sm font-normal leading-normal transition-colors" href="#">服務條款</a>
            <a className="text-white/50 hover:text-white/80 text-sm font-normal leading-normal transition-colors" href="#">隱私權政策</a>
          </div>
      </footer>
    </div>
  );
}