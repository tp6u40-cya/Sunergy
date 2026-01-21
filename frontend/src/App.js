// src/App.js
import React, { useEffect, useState } from 'react';
import PublicHome from './pages/PublicHome';
import LoginModal from './components/LoginModal';
import RegisterModal from './components/RegisterModal';
import Dashboard from './pages/Dashboard';
import DataVariableGuide from './pages/DataVariableGuide';
import StartPredict from './pages/StartPredict';
import DataCleaning from './pages/DataCleaning';
import UnitAdjustment from './pages/UnitAdjustment';
import ModelTraining from './pages/ModelTraining';
import PredictSolar from './pages/PredictSolar'; // 確保檔案名稱與組件對應正確
import Sites from './pages/Sites';
import ModelManagement from './pages/ModelManagement';
import UserGuide from './pages/UserGuide';
import CreateSiteModal from './components/CreateSiteModal';

function App() {
  const [isLoginModalOpen, setIsLoginModalOpen] = useState(false);
  const [isRegisterModalOpen, setIsRegisterModalOpen] = useState(false);
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [currentPage, setCurrentPage] = useState('home'); 
  const [isCreateSiteModalOpen, setIsCreateSiteModalOpen] = useState(false);

  // Restore login from JWT on refresh (minimal, non-breaking)
  useEffect(() => {
    const restore = async () => {
      try {
        const stored = JSON.parse(localStorage.getItem('user') || '{}');
        const token = stored?.access_token;
        if (token) {
          const res = await fetch('http://127.0.0.1:8000/auth/me', {
            headers: { Authorization: `Bearer ${token}` },
          });
          if (res.ok) {
            setIsLoggedIn(true);
            return;
          }
          // token invalid; clear it
          localStorage.removeItem('user');
          localStorage.removeItem('user_id');
        } else if (stored?.user_id) {
          // Backward compatibility: if prior login stored user info only
          setIsLoggedIn(true);
        }
      } catch {
        // ignore
      }
    };
    restore();
  }, []);

  // --- 認證與導航函式 ---
  const handleOpenLogin = () => { setIsRegisterModalOpen(false); setIsLoginModalOpen(true); };
  const handleOpenRegister = () => { setIsLoginModalOpen(false); setIsRegisterModalOpen(true); };
  const handleCloseAuthModals = () => { setIsLoginModalOpen(false); setIsRegisterModalOpen(false); };

  const handleLoginSuccess = () => {
    setIsLoggedIn(true);
    handleCloseAuthModals();
    setCurrentPage('dashboard');
  };

  const handleLogout = () => { setIsLoggedIn(false); setCurrentPage('home'); };

  const handleGoToDashboard = () => { setCurrentPage('dashboard'); window.scrollTo(0, 0); };
  const handleGoToTrainFlow = () => { setCurrentPage('data-guide'); window.scrollTo(0, 0); }; 
  const handleGoToPredictSolar = () => { setCurrentPage('predict-solar'); window.scrollTo(0, 0); };
  const handleGoToSites = () => { setCurrentPage('site'); window.scrollTo(0, 0); };
  
  const handleGoToModelMgmt = () => { setCurrentPage('model-mgmt'); window.scrollTo(0, 0); };

  // --- 教學與 Modal 控制 ---
  const handleOpenUserGuide = () => { setCurrentPage('user-guide'); };
  const handleFinishUserGuide = () => { setCurrentPage('home'); };
  const handleOpenCreateSite = () => setIsCreateSiteModalOpen(true);
  const handleCloseCreateSite = () => setIsCreateSiteModalOpen(false);
  
  // 以最小變動方式：在父層處理建立站點 API 呼叫
  const handleSubmitCreateSiteApi = async (form) => {
    try {
      const user = JSON.parse(localStorage.getItem('user') || '{}');
      const uid = user?.user_id;
      const token = user?.access_token;
      if (!uid) {
        alert('請先登入後再建立站點');
        setIsCreateSiteModalOpen(false);
        return;
      }

      const payload = {
        site_code: form.site_code,
        site_name: form.site_name,
        location: form.location,
        user_id: uid,
      };

      const res = await fetch('http://127.0.0.1:8000/site/create', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok) {
        alert(data?.detail || '建立站點失敗');
      } else {
        alert(`已建立站點：${form.site_name}`);
      }
    } catch (e) {
      alert('建立站點時發生錯誤');
    } finally {
      setIsCreateSiteModalOpen(false);
    }
  };
  
  const handleSubmitCreateSite = (data) => {
    alert(`已成功建立案場：${data.name}`);
    setIsCreateSiteModalOpen(false);
  };

  const renderContent = () => {
    if (currentPage === 'user-guide') return <UserGuide onFinish={handleFinishUserGuide} />;

    if (!isLoggedIn) {
      return (
        <>
          <PublicHome onOpenLogin={handleOpenLogin} onOpenUserGuide={handleOpenUserGuide} />
          {isLoginModalOpen && <LoginModal onClose={handleCloseAuthModals} onSwitchToRegister={handleOpenRegister} onLoginSuccess={handleLoginSuccess} />}
          {isRegisterModalOpen && <RegisterModal onClose={handleCloseAuthModals} onSwitchToLogin={handleOpenLogin} />}
        </>
      );
    }

    const commonNavbarProps = {
      activePage: currentPage,
      onNavigateToDashboard: handleGoToDashboard,
      onNavigateToTrain: handleGoToTrainFlow,
      onNavigateToPredict: handleGoToPredictSolar,
      onNavigateToSites: handleGoToSites,
      onNavigateToModelMgmt: handleGoToModelMgmt,
      onLogout: handleLogout
    };

    switch (currentPage) {
      case 'dashboard':
        return <Dashboard {...commonNavbarProps} onOpenCreateSite={handleOpenCreateSite} />;
      
      case 'site':
        return <Sites {...commonNavbarProps} onOpenCreateSite={handleOpenCreateSite} />;
      
      case 'predict-solar':
        return <PredictSolar {...commonNavbarProps} onBack={handleGoToDashboard} />;

      case 'model-mgmt': 
        return <ModelManagement {...commonNavbarProps} />;

      /* --- 訓練流程五步驟 --- */

      case 'data-guide': // (1) 變數說明
        return <DataVariableGuide {...commonNavbarProps} 
                onBack={handleGoToDashboard} 
                onNext={() => setCurrentPage('start-predict')} />;

      case 'start-predict': // (2) 上傳資料
        return <StartPredict {...commonNavbarProps} 
                onBack={() => setCurrentPage('data-guide')} 
                onNext={() => setCurrentPage('data-cleaning')} />;
      
      case 'data-cleaning': // (3) 清理資料
        return <DataCleaning {...commonNavbarProps} 
                onBack={() => setCurrentPage('start-predict')} 
                onNext={() => setCurrentPage('unit-adjustment')} />;
      
      case 'unit-adjustment': // (4) 調整單位
        return <UnitAdjustment {...commonNavbarProps} 
                onBack={() => setCurrentPage('data-cleaning')} 
                onNext={() => setCurrentPage('model-training')} />;
      
      case 'model-training': // (5) 模型訓練與優化
        return <ModelTraining {...commonNavbarProps} 
                onBack={() => setCurrentPage('unit-adjustment')} 
                onNext={handleGoToPredictSolar} />;

      default:
        return <Dashboard {...commonNavbarProps} onOpenCreateSite={handleOpenCreateSite} />;
    }
  };

  return (
    <>
      {renderContent()}
      {isLoggedIn && isCreateSiteModalOpen && (
        <CreateSiteModal onClose={handleCloseCreateSite} onSubmit={handleSubmitCreateSiteApi} />
      )}
    </>
  );
}

export default App;
