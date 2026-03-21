import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    Cpu,
    BarChart3,
    Play,
    RefreshCcw,
    ChevronRight,
    ShieldCheck,
    Activity,
    Trees,
    X
} from 'lucide-react';
import './App.css';

const App = () => {
    const [loading, setLoading] = useState(false);
    const [status, setStatus] = useState({ is_training: false, logs: [], has_results: false, results: null, plot_url: null });
    const [hasClickedTrain, setHasClickedTrain] = useState(false);
    const [viewerUrl, setViewerUrl] = useState(null);
    const [showAbout, setShowAbout] = useState(false);

    useEffect(() => {
        const interval = setInterval(async () => {
            try {
                const res = await fetch('/api/status');
                const data = await res.json();
                setStatus({
                    is_training: data.is_training || false,
                    logs: data.logs || [],
                    has_results: data.has_results || false,
                    results: data.results || null,
                    plot_url: data.plot_url || null
                });
                if (!data.is_training && loading) {
                    setLoading(false);
                }
            } catch (e) {
                console.error("Failed to fetch status", e);
            }
        }, 1000);
        return () => clearInterval(interval);
    }, [loading]);

    const handleTrain = async () => {
        setHasClickedTrain(true);
        setLoading(true);
        await fetch('/api/train', { method: 'POST' });
    };

    const ReportTable = ({ report }) => {
        if (!report) return null;
        const classes = ["0", "1", "2", "3"];
        const labels = ["World", "Sports", "Business", "Sci/Tech"];

        return (
            <div className="report-table-container">
                <table className="report-table">
                    <thead>
                        <tr>
                            <th>Class</th>
                            <th>Prec.</th>
                            <th>Recall</th>
                            <th>F1</th>
                        </tr>
                    </thead>
                    <tbody>
                        {classes.map((cls, idx) => (
                            <tr key={cls}>
                                <td>{labels[idx]}</td>
                                <td>{report[cls]?.precision?.toFixed(2) || "0.00"}</td>
                                <td>{report[cls]?.recall?.toFixed(2) || "0.00"}</td>
                                <td>{report[cls]?.["f1-score"]?.toFixed(2) || "0.00"}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        );
    };

    return (
        <div className="app-container">
            <nav className="navbar">
                <div className="logo">
                    <Cpu className="logo-icon" />
                    <span>NeuroNews AI</span>
                </div>
                <div className="nav-links">
                    <button className="nav-link-btn" onClick={() => setViewerUrl("/api/files/COMP338_Project2.pdf")}>Project Description</button>
                    <button className="nav-link-btn" onClick={() => setShowAbout(true)}>About Us</button>
                    <button className="nav-link-btn" onClick={() => setViewerUrl("/api/files/AICourseProject2_Report.pdf")}>Documentation</button>
                </div>
            </nav>

            <main className="content">
                <header className="hero">
                    <motion.h1
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.6 }}
                    >
                        Classifying the World’s <span>Information</span>
                    </motion.h1>
                    <motion.p
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.6, delay: 0.1 }}
                    >
                        Empower your news discovery with dual-algorithmic sentiment and category analysis.
                        Train on AG News dataset and visualize decision boundaries in real-time.
                    </motion.p>

                    <motion.div
                        className="action-container"
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ duration: 0.6, delay: 0.2 }}
                    >
                        <button
                            className={`train-btn ${status.is_training ? 'training' : ''}`}
                            onClick={handleTrain}
                            disabled={status.is_training}
                        >
                            {status.is_training ? (
                                <>
                                    <RefreshCcw className="icon spin" />
                                    Neural Nodes Connecting...
                                </>
                            ) : (
                                <>
                                    <Play className="icon" />
                                    Initialize Training Sequence
                                </>
                            )}
                        </button>
                    </motion.div>
                    <AnimatePresence>
                        {hasClickedTrain && status.is_training && (
                            <motion.div
                                className="terminal-loader"
                                initial={{ opacity: 0, height: 0 }}
                                animate={{ opacity: 1, height: 'auto' }}
                                exit={{ opacity: 0, height: 0 }}
                            >
                                <div className="terminal-header">
                                    <div className="dot red"></div>
                                    <div className="dot yellow"></div>
                                    <div className="dot green"></div>
                                    <span>system_logs — training_v1.0.4</span>
                                </div>
                                <div className="terminal-body">
                                    {status.logs.map((log, i) => (
                                        <motion.div
                                            key={i}
                                            className="log-line"
                                            initial={{ opacity: 0, x: -10 }}
                                            animate={{ opacity: 1, x: 0 }}
                                        >
                                            <span className="prompt">{">"}</span> {log}
                                        </motion.div>
                                    ))}
                                    <div className="cursor-line">
                                        <span className="prompt">{">"}</span>
                                        <span className="cursor">|</span>
                                    </div>
                                </div>
                            </motion.div>
                        )}
                    </AnimatePresence>
                </header>

                <AnimatePresence>
                    {hasClickedTrain && status.has_results && !status.is_training && (
                        <motion.section
                            className="results-grid"
                            initial={{ opacity: 0, y: 40 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0 }}
                        >
                            <div className="metric-card glass">
                                <div className="card-header">
                                    <ShieldCheck className="icon purple" />
                                    <h3>Logistic Regression</h3>
                                </div>
                                <div className="accuracy-val">
                                    {status.results?.logistic_regression?.accuracy
                                        ? (status.results.logistic_regression.accuracy * 100).toFixed(1) + "%"
                                        : "N/A"}
                                    <span>Accuracy Score</span>
                                </div>
                                <ReportTable report={status.results?.logistic_regression?.report} />
                            </div>

                            <div className="metric-card glass">
                                <div className="card-header">
                                    <Activity className="icon rose" />
                                    <h3>Decision Tree</h3>
                                </div>
                                <div className="accuracy-val">
                                    {status.results?.decision_tree?.accuracy
                                        ? (status.results.decision_tree.accuracy * 100).toFixed(1) + "%"
                                        : "N/A"}
                                    <span>Accuracy Score</span>
                                </div>
                                <ReportTable report={status.results?.decision_tree?.report} />
                            </div>

                            <div className="visual-card glass span-2">
                                <div className="card-header">
                                    <Trees className="icon blue" />
                                    <h3>Neural Architecture Mapping</h3>
                                </div>
                                <div className="tree-plot-container">
                                    <img src={status.plot_url} alt="Decision Tree" className="tree-plot" />
                                </div>
                            </div>

                        </motion.section>
                    )}
                </AnimatePresence>
            </main>

            <footer className="footer">
                <p>&copy; 2026 Made for AI course "Artificial Intelligence (COMP338)"</p>
            </footer>

            <AnimatePresence>
                {viewerUrl && (
                    <motion.div
                        className="modal-overlay"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                    >
                        <motion.div
                            className="modal-content glass"
                            initial={{ scale: 0.9, opacity: 0 }}
                            animate={{ scale: 1, opacity: 1 }}
                            exit={{ scale: 0.9, opacity: 0 }}
                        >
                            <div className="modal-header">
                                <h3>Document Viewer</h3>
                                <button className="close-btn" onClick={() => setViewerUrl(null)}>
                                    <X size={18} />
                                    <span>Close Viewer</span>
                                </button>
                            </div>
                            <iframe
                                src={viewerUrl}
                                className="pdf-frame"
                                title="PDF Viewer"
                            />
                        </motion.div>
                    </motion.div>
                )}
            </AnimatePresence>

            <AnimatePresence>
                {showAbout && (
                    <motion.div
                        className="modal-overlay"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                    >
                        <motion.div
                            className="about-modal glass"
                            initial={{ y: 50, opacity: 0 }}
                            animate={{ y: 0, opacity: 1 }}
                            exit={{ y: 50, opacity: 0 }}
                        >
                            <div className="modal-header">
                                <h3>Research Laboratory Team</h3>
                                <button className="close-btn" onClick={() => setShowAbout(false)}>
                                    <X size={18} />
                                    <span>Close</span>
                                </button>
                            </div>

                            <div className="team-grid">
                                <div className="team-member glass">
                                    <div className="member-photo">
                                        <img src="/team/qais.png" alt="Qais Alqrem" />
                                    </div>
                                    <h4>Qais Alqrem</h4>
                                    <p className="role">Machine Learning Engineer</p>
                                    <p className="id">ID: 1211791</p>
                                </div>

                                <div className="team-member glass">
                                    <div className="member-photo">
                                        <img src="/team/tala.jpg" alt="Tala Kherawish" />
                                    </div>
                                    <h4>Tala Kherawish</h4>
                                    <p className="role">Chief AI Research Scientist & Lead Architect</p>
                                    <p className="id">ID: 1220536</p>
                                </div>

                                <div className="team-member glass">
                                    <div className="member-photo">
                                        <img src="/team/rand.png" alt="Rand Tabakhna" />
                                    </div>
                                    <h4>Rand Tabakhna</h4>
                                    <p className="role">Neural Data Specialist & Analyst</p>
                                    <p className="id">ID: 1222845</p>
                                </div>
                            </div>
                        </motion.div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};

export default App;
