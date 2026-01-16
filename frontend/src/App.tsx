import React, { useState } from 'react';
import axios from 'axios';
import { Loader2, ArrowRight, CheckCircle2 } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import logo from './assets/logo.png';
import BioForm from './components/BioForm';
import type { BioData } from './components/BioForm';
import ImageUpload from './components/ImageUpload';
import Results from './components/Results';

function App() {
  const [step, setStep] = useState(1);
  const [bioData, setBioData] = useState<BioData | null>(null);
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [analysisResult, setAnalysisResult] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleBioSubmit = (data: BioData) => {
    setBioData(data);
    setStep(2);
  };

  const handleImageSelect = (file: File) => {
    setSelectedImage(file);
  };

  const handleAnalyze = async () => {
    if (!selectedImage || !bioData) return;
    
    setIsLoading(true);
    const formData = new FormData();
    formData.append('file', selectedImage);
    
    try {
      // 1. Get Prediction
      const res = await axios.post('http://localhost:8000/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setAnalysisResult(res.data);
      setStep(3);
    } catch (err) {
      console.error(err);
      alert("Analysis failed. Please check backend connection.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleDownloadPdf = async () => {
    if (!analysisResult || !selectedImage || !bioData) return;

    const formData = new FormData();
    formData.append('class_name', analysisResult.class);
    formData.append('confidence', analysisResult.confidence);
    formData.append('heatmap_b64', analysisResult.heatmap);
    formData.append('original_img', selectedImage);
    formData.append('language', 'en'); // Default to English for now or add selector
    
    // Add Bio Data to Form
    Object.entries(bioData).forEach(([key, value]) => {
        formData.append(key, value);
    });

    try {
      const res = await axios.post('http://localhost:8000/generate_report', formData, {
        responseType: 'blob'
      });
      
      const url = window.URL.createObjectURL(new Blob([res.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `MedOral_Report_${Date.now()}.pdf`);
      document.body.appendChild(link);
      link.click();
    } catch (err) {
      console.error("PDF Download failed", err);
      alert("Failed to generate PDF.");
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 font-sans selection:bg-blue-100">
      
      {/* Header */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-10 shadow-sm">
        <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <img src={logo} alt="MedOral AI" className="h-10 w-auto object-contain" />
            <span className="font-bold text-xl tracking-tight text-slate-800">MedOral<span className="text-blue-600">AI</span></span>
          </div>
          <div className="text-sm font-medium text-slate-500 flex items-center gap-6">
            <div className={`flex items-center gap-2 ${step >= 1 ? 'text-blue-600' : ''}`}>
              <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs border ${step >= 1 ? 'border-blue-600 bg-blue-50' : 'border-slate-300'}`}>1</div>
              Assessment
            </div>
             <div className="w-8 h-px bg-slate-200"></div>
            <div className={`flex items-center gap-2 ${step >= 2 ? 'text-blue-600' : ''}`}>
              <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs border ${step >= 2 ? 'border-blue-600 bg-blue-50' : 'border-slate-300'}`}>2</div>
              Scan
            </div>
             <div className="w-8 h-px bg-slate-200"></div>
            <div className={`flex items-center gap-2 ${step >= 3 ? 'text-blue-600' : ''}`}>
              <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs border ${step >= 3 ? 'border-blue-600 bg-blue-50' : 'border-slate-300'}`}>3</div>
              Results
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-6xl mx-auto px-6 py-12 min-h-[calc(100vh-160px)]">
        <AnimatePresence mode="wait">
        {step === 1 && (
          <motion.div 
            key="step1"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 20 }}
            transition={{ duration: 0.3 }}
          >
            <div className="text-center mb-10">
              <h1 className="text-3xl font-bold text-slate-900 mb-3">Patient Risk Assessment</h1>
              <p className="text-slate-500 max-w-lg mx-auto">Please answer a few questions about the patient's history to improve the analysis accuracy.</p>
            </div>
            <BioForm onComplete={handleBioSubmit} />
          </motion.div>
        )}

        {step === 2 && (
          <motion.div 
            key="step2"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 20 }}
            transition={{ duration: 0.3 }}
          >
             <div className="text-center mb-10">
              <h1 className="text-3xl font-bold text-slate-900 mb-3">Upload Oral Scan</h1>
              <p className="text-slate-500 max-w-lg mx-auto">Upload a clear, well-lit image of the oral cavity or the specific lesion.</p>
            </div>
            
            <div className="bg-white p-8 rounded-2xl shadow-sm border border-slate-100 max-w-xl mx-auto">
              <ImageUpload onImageSelect={handleImageSelect} />
              
              {selectedImage && (
                <div className="mt-8">
                  <button 
                    onClick={handleAnalyze} 
                    disabled={isLoading}
                    className="w-full bg-primary hover:bg-primary/90 text-primary-foreground font-semibold py-4 rounded-xl transition-all flex items-center justify-center gap-2 shadow-lg disabled:opacity-70 disabled:cursor-not-allowed"
                  >
                    {isLoading ? (
                      <>
                        <Loader2 className="animate-spin" /> Analyzing with DenseNet201...
                      </>
                    ) : (
                      <>
                        Run AI Analysis <ArrowRight size={20} />
                      </>
                    )}
                  </button>
                </div>
              )}
            </div>
            
            <div className="mt-6 text-center">
                <button onClick={() => setStep(1)} className="text-slate-400 hover:text-slate-600 text-sm">
                    ← Back to Assessment
                </button>
            </div>
          </motion.div>
        )}

        {step === 3 && analysisResult && (
          <motion.div 
            key="step3"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 20 }}
            transition={{ duration: 0.3 }}
          >
             <div className="text-center mb-10">
              <h1 className="text-3xl font-bold text-slate-900 mb-3">Analysis Complete</h1>
              <div className="flex items-center justify-center gap-2 text-green-600 bg-green-50 w-fit mx-auto px-4 py-1 rounded-full text-sm font-medium">
                <CheckCircle2 size={16} /> Processed successfully in 1.2s
              </div>
            </div>
            <Results data={analysisResult} onDownloadPdf={handleDownloadPdf} bioData={bioData} />
             <div className="mt-12 text-center">
                <button 
                    onClick={() => { setStep(1); setBioData(null); setSelectedImage(null); setAnalysisResult(null); }}
                    className="text-primary hover:text-primary/80 font-medium"
                >
                    Start New Analysis
                </button>
            </div>
          </motion.div>
        )}
        </AnimatePresence>
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-slate-200 py-8 mt-auto">
        <div className="max-w-6xl mx-auto px-6 text-center text-slate-500 text-sm">
            <p className="mb-2">&copy; {new Date().getFullYear()} MedOral AI. All rights reserved.</p>
            <p className="text-xs text-slate-400">
                This tool is for educational and experimental purposes only. It is not a substitute for professional medical diagnosis.
            </p>
        </div>
      </footer>
    </div>
  );
}

export default App;