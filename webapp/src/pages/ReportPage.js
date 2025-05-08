import { useLocation, useNavigate } from 'react-router-dom';
import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Tooltip
} from 'recharts';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, ResponsiveContainer } from 'recharts';
import { PieChart, Pie, Cell, Legend } from 'recharts';
import { ReactComponent as Mansvg } from '../assets/mansvgrepo.svg';

import { useReactToPrint } from 'react-to-print';
import { FaDownload } from 'react-icons/fa';
import { useRef } from 'react';



export default function ReportPage() {
  const location = useLocation();
  const navigate = useNavigate();


  const reportData = location.state?.reportData;
  const username = 'demo';
  const date = "today";

  const reportRef = useRef(null);

  const handlePrint = useReactToPrint({
    // NEW API — just give it the ref object
    contentRef: reportRef,
    copyStyles: true,
    documentTitle: 'Performance Report',
  });

  if (!reportData) {
    navigate('/');
    return null;
  }

  const {
    emotionScore,
    gazeScore,
    movementScore,
    shoulderScore,
    handsScore,
    speechScore,
    overallScore,
    overallSummary
  } = reportData;

  const maxSecond = Math.max(...Object.keys(reportData.emotion).map(Number));

  const emotionPerSecond = Array.from({ length: maxSecond + 1 }, (_, sec) =>
    reportData.emotion[sec] || 'None'
  );

  const emotionColors = {
    Neutral: '#cccccc',
    Happy: '#ffd700',
    Sad: '#1e90ff',
    Surprise: '#ff6347',
    Contempt: '#8b008b',
    Anger: '#ff4500',
    Disgust: '#228b22',
    None: '#ffffff'  // fallback for missing
  };

  const gazePerSecond = Array.from({ length: maxSecond + 1 }, (_, sec) =>
    reportData.gaze[sec] || 'None'
  );
  const gazeCounts = gazePerSecond.reduce((acc, val) => {
    const cleaned = val.trim().toLowerCase();
    acc[cleaned] = (acc[cleaned] || 0) + 1;
    return acc;
  }, {});
  const total = gazePerSecond.length;
  const getHeatColor = (percent) => {
    const baseAlpha = 0.1;
    const extraAlpha = Math.min(0.7, percent / 100); // stronger purple as percentage increases
    return `rgba(107, 76, 175, ${baseAlpha + extraAlpha})`;
  };
  const gazePercentages = {
    'up left': ((gazeCounts['up left'] || 0) / total * 100).toFixed(1),
    'up': ((gazeCounts['up'] || 0) / total * 100).toFixed(1),
    'up right': ((gazeCounts['up right'] || 0) / total * 100).toFixed(1),
    'left': ((gazeCounts['left'] || 0) / total * 100).toFixed(1),
    'center': ((gazeCounts['straight'] || 0) / total * 100).toFixed(1),
    'right': ((gazeCounts['right'] || 0) / total * 100).toFixed(1),
    'down left': ((gazeCounts['down left'] || 0) / total * 100).toFixed(1),
    'down': ((gazeCounts['down'] || 0) / total * 100).toFixed(1),
    'down right': ((gazeCounts['down right'] || 0) / total * 100).toFixed(1)
  };

  const shoulderPerSecond = Array.from({ length: maxSecond + 1 }, (_, sec) =>
    reportData.shoulder[sec] || 'None'
  );

  const handsPerSecond = Array.from({ length: maxSecond + 1 }, (_, sec) =>
    reportData.gesture[sec] || 'None'
  );
  const toPercentageData = (arr) => {
    const counts = arr.reduce((acc, val) => {
      const cleaned = val.replace('°', '').trim();
      acc[cleaned] = (acc[cleaned] || 0) + 1;
      return acc;
    }, {});
    const total = arr.length;
    return Object.entries(counts).map(([name, count]) => ({
      name,
      value: Number(((count / total) * 100).toFixed(1)) // ensure it's a number
    }));
  };

  const shoulderData = toPercentageData(shoulderPerSecond);
  const handsData = toPercentageData(handsPerSecond);

  const pieColors = ['#6b4caf', '#82ca9d', '#8884d8', '#ffc658'];


  const movementPerSecond = Array.from({ length: maxSecond + 1 }, (_, sec) =>
    reportData.movement[sec] !== undefined ? reportData.movement[sec] : 0
  );

  const movementData = movementPerSecond.map((pos, index) => ({
    second: index,
    position: pos,
    label:
      pos <= 2
        ? 'Left'
        : pos <= 4
          ? 'Middle Left'
          : pos <= 6
            ? 'Center'
            : pos <= 8
              ? 'Middle Right'
              : 'Right'
  }));

  let rank = 'E';
  if (overallScore >= 90) rank = 'S';
  else if (overallScore >= 80) rank = 'A';
  else if (overallScore >= 70) rank = 'B';
  else if (overallScore >= 60) rank = 'C';
  else if (overallScore >= 50) rank = 'D';

  const radarData = [
    { subject: 'Emotion', A: emotionScore, fullMark: 100 },
    { subject: 'Gaze', A: gazeScore, fullMark: 100 },
    { subject: 'Movement', A: movementScore, fullMark: 100 },
    { subject: 'Shoulder', A: shoulderScore, fullMark: 100 },
    { subject: 'Gesture', A: handsScore, fullMark: 100 },
    { subject: 'Speech', A: speechScore, fullMark: 100 }
  ];

  const styles = {
    container: {
      backgroundColor: '#f9f9f9',
      minHeight: '100vh',
      padding: '20px',
      fontFamily: '"Segoe UI", Tahoma, Geneva, Verdana, sans-serif'
    },
    reportBoxOuter: {
      backgroundColor: '#6b4caf',
      borderRadius: '20px',
      padding: '10px',
      maxWidth: '1200px',
      minWidth: '600px',
      margin: '0 auto'
    },
    reportBoxInner: {
      border: '2px dotted white',
      borderRadius: '15px',
      padding: '20px'
    },
    title: {
      textAlign: 'center',
      fontSize: '36px',
      marginBottom: '20px',
      color: 'white',
      fontWeight: '600',
      letterSpacing: '1px'
    },
    divider: {
      borderTop: '2px solid white',
      margin: '20px 0'
    },
    reportContainer: {
      display: 'flex',
      flexWrap: 'wrap',
      gap: '20px',
      justifyContent: 'center'
    },
    leftPanel: {
      backgroundColor: '#fff',
      borderRadius: '15px',
      padding: '40px',
      minWidth: '300px',
      justifyContent: 'center',
      justifyItems: 'center',
      flex: '1 1 400px'
    },
    rightPanel: {
      display: 'flex',
      flexDirection: 'column',
      gap: '20px',
      flex: '1 1 300px'
    },
    card: {
      backgroundColor: 'white',
      color: '#6b4caf',
      borderRadius: '15px',
      padding: '20px',
      textAlign: 'center'
    },
    cardTitle: {
      fontSize: '12px',
      textTransform: 'uppercase',
      fontWeight: '500'
    },
    cardValue: {
      fontSize: '50px',
      fontWeight: '1000',
      fontStyle: 'italic'
    },
    userInfoCard: {
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      padding: '20px'
    },
    userInfoValue: {
      fontSize: '18px',
      fontWeight: '700',
      marginBottom: '10px'
    },
    summarySection: {
      backgroundColor: 'white',
      color: '#6b4caf',
      borderRadius: '15px',
      padding: '20px',
      textAlign: 'start',
      fontSize: '18px',
      fontWeight: '500',
      lineHeight: '1.5',
      marginTop: '20px',
      wordWrap: 'break-word',
      overflowWrap: 'break-word'
    },
    textSection:{
      backgroundColor: 'white',
      color: '#6b4caf',
      borderRadius: '15px',
      padding: '0px',
      textAlign: 'start',
      fontSize: '18px',
      fontWeight: '500',
      lineHeight: '1.5',
      marginTop: '0px',
      wordWrap: 'break-word',
      overflowWrap: 'break-word'
    },
    summaryTitle: {
      fontSize: '22px',
      fontWeight: '700',
      marginBottom: '10px'
    },
    breakdownSection: {
      backgroundColor: 'white',
      color: '#6b4caf',
      borderRadius: '15px',
      padding: '20px',
      marginTop: '20px',
      textAlign: 'start',
      fontSize: '18px',
      fontWeight: '500',
      lineHeight: '1.5'
    },
    breakdownContent: {
      display: 'flex',
      flexDirection: 'column',
      gap: '20px',
      marginTop: '10px'
    },
    placeholderBox: {
      backgroundColor: '#f1f1f1',
      borderRadius: '10px',
      padding: '15px',
      textAlign: 'center',
    },
    breakdownTitle: {
      fontSize: '26px',
      fontWeight: '700',
      textAlign: 'center',
      marginBottom: '20px'
    },
    graphSection: {
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      marginBottom: '30px'
    },
    graphTitle: {
      fontSize: '20px',
      fontWeight: '600',
      marginBottom: '10px'
    },
    graphBar: {
      display: 'flex',
      width: '100%', // or '100%' or any fixed width you want
      height: '60px', // match block height
      overflow: 'hidden',
      borderRadius: '10px'
    },
    graphBlockContainer: {
      margin: '0', // remove margin between blocks
      padding: '0'
    },
    graphBlock: {
      flex: '1',
      height: '60px', // taller bar
      transition: 'background-color 0.3s ease'
    },
    graphBlockLabel: {
      fontSize: '8px',
      color: '#333'
    },
    suggestionsTitle: {
      fontSize: '16px',
      fontWeight: '600',
      marginTop: '10px',
      marginBottom: '5px',
      textAlign: 'left',
      textDecoration: 'underline'
    },
    legendContainer: {
      display: 'flex',
      justifyContent: 'center',
      flexWrap: 'wrap',
      marginTop: '10px',
      gap: '10px'
    },
    legendItem: {
      display: 'flex',
      alignItems: 'center',
      fontSize: '12px'
    },
    legendColor: {
      width: '12px',
      height: '12px',
      borderRadius: '2px',
      marginRight: '5px',
      border: '1px solid #ccc'
    },
    legendLabel: {
      fontSize: '12px',
      color: '#333'
    },
    suggestionText: {
      fontSize: '14px',
      color: '#555',
      textAlign: 'left',
      alignSelf: 'flex-aSTART'
    },
    barLabels: {
      display: 'flex',
      justifyContent: 'space-between',
      fontSize: '10px',
      marginTop: '4px',
      padding: '0 4px'
    },
    barLabel: {
      color: '#555'
    }
  };

  function getGradientString(emotionArray, emotionColors) {
    let stops = [];
    const step = 100 / emotionArray.length;

    for (let i = 0; i < emotionArray.length; i++) {
      const color = emotionColors[emotionArray[i]] || '#000';
      const percent = i * step;
      stops.push(`${color} ${percent}%`);
    }
    // Add final stop at 100%
    stops.push(`${emotionColors[emotionArray[emotionArray.length - 1]] || '#000'} 100%`);

    return `linear-gradient(to right, ${stops.join(', ')})`;
  }

  const gradientString = getGradientString(emotionPerSecond, emotionColors);


  return (
    <div style={styles.container}>
      <div >
        <div ref={reportRef} style={styles.reportBoxOuter}>
          <div style={styles.reportBoxInner}>
            <h1 style={styles.title}>Here Are Your Results!</h1>
            <div style={styles.divider}></div>

            <div className="report-container" style={styles.reportContainer}>
              {/* LEFT PANEL */}
              <div className="left-panel" style={styles.leftPanel}>
                <div style={{ textAlign: 'center', marginBottom: '10px', color: '#6b4caf', fontSize: '20px', fontWeight: '600' }}>
                  Performance Radar
                </div>
                <RadarChart width={400} height={300} cx={200} cy={150} outerRadius={120} data={radarData}>
                  <PolarGrid stroke="#6b4caf" />
                  <PolarAngleAxis dataKey="subject" stroke="#6b4caf" />
                  <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
                  <Radar name="Score" dataKey="A" stroke="#6b4caf" fill="#6b4caf" fillOpacity={0.6} isAnimationActive={true} />
                  <Tooltip />
                </RadarChart>
              </div>

              {/* RIGHT PANEL */}
              <div className="right-panel" style={styles.rightPanel}>
                <div style={styles.card}>
                  <div style={styles.cardTitle}>Score</div>
                  <div style={styles.cardValue}>{overallScore}%</div>
                </div>
                <div style={styles.card}>
                  <div style={styles.cardTitle}>Rank</div>
                  <div style={styles.cardValue}>{rank}</div>
                </div>
                <div style={{ ...styles.card, ...styles.userInfoCard }}>
                  <div style={styles.cardTitle}>Username</div>
                  <div style={styles.userInfoValue}>{username}</div>
                  <div style={styles.cardTitle}>Date</div>
                  <div style={styles.userInfoValue}>{date}</div>
                </div>
              </div>
            </div>

            {/* OVERVIEW SECTION */}
            <div style={styles.summarySection}>
              <div style={styles.summaryTitle}>Overview</div>
              <div>{overallSummary}</div>
            </div>

            <div style={styles.divider}></div>
            <h1 style={styles.title}>Speech Analysis</h1>
            {/* TRANSCRIPT & SPEECH IMPROVEMENT SECTIONS */}
            <div style={styles.breakdownSection}>
              <div style={styles.summaryTitle}>Transcript</div>
              <div style={styles.textSection}>
                {reportData.transcriptSegments
                  .split('\n')
                  .map((line, idx) => {
                    // pull out “[0.00s - 6.70s]” vs the rest
                    const m = line.match(/^\[(.*?)\]\s*(.*)/) || [];
                    const timestamp = m[1] || line;
                    const text = m[2] || '';
                    return (
                      <div key={idx} style={{ marginBottom: '1px' }}>
                        <span style={{ color: '#6b4caf', fontWeight: 400 }}>
                          [{timestamp}]
                        </span>{' '}
                        <span style={{ color: '#000', fontWeight: 200 }}>
                          {text}
                        </span>
                      </div>
                    );
                  })}
              </div>
            </div>
            <div style={styles.breakdownSection}>
              <div style={styles.summaryTitle}>Speech Improvement Assistance</div>
              <div style={styles.suggestionText}>
                {reportData.speechImprovements}
              </div>
            </div>
            <div style={styles.divider}></div>
            {/* SCORE BREAKDOWN SECTION */}
            <div style={styles.breakdownSection}>
              <div style={styles.breakdownTitle}>Score Breakdown</div>

              <div style={styles.breakdownContent}>
                {/* Face Emotion Analysis */}
                <div style={styles.placeholderBox}>
                  <div style={styles.graphTitle}>Face Emotion Analysis</div>
                  <div style={{
                    ...styles.graphBar,
                    background: gradientString
                  }}>
                    {emotionPerSecond.map((emotion, index) => (
                      <div
                        key={index}
                        title={`Time: ${index}: ${emotion}`}
                        style={{
                          ...styles.graphBlock,
                          backgroundColor: emotionColors[emotion] || '#000',
                          cursor: 'pointer'
                        }}
                      />
                    ))}
                  </div>
                  {/* Video Start / End labels */}
                  <div style={styles.barLabels}>
                    <span style={styles.barLabel}>Video Start</span>
                    <span style={styles.barLabel}>Video End</span>
                  </div>

                  <div style={styles.legendContainer}>
                    {Object.entries(emotionColors).map(([emotion, color]) => (
                      <div key={emotion} style={styles.legendItem}>
                        <div style={{ ...styles.legendColor, backgroundColor: color }} />
                        <div style={styles.legendLabel}>{emotion}</div>
                      </div>
                    ))}
                  </div>

                  <div style={styles.suggestionsTitle}>Suggestions for Improvement</div>
                  <div style={styles.suggestionText}>{reportData.emotionText}</div>
                </div>

                {/* Placeholder Graph 2 */}
                <div style={styles.placeholderBox}>
                  <div style={styles.graphTitle}>Movement Analysis</div>
                  <div style={{ width: '100%', height: 300 }}>
                    <ResponsiveContainer>
                      <LineChart data={movementData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis
                          dataKey="second"
                          tickFormatter={(val, index) => {
                            if (index === 2) return 'Video Start';
                            if (index === movementData.length - 6) return 'Video End';
                            return '';
                          }}
                        />
                        <YAxis
                          domain={[0, 10]}
                          ticks={[1, 3, 5, 7, 9]}
                          tickFormatter={(val) => {
                            if (val === 1) return 'Left';
                            if (val === 3) return 'Middle Left';
                            if (val === 5) return 'Center';
                            if (val === 7) return 'Middle Right';
                            if (val === 9) return 'Right';
                            return '';
                          }}
                        />
                        <Tooltip
                          labelFormatter={(label) => `Time: ${label}`}
                          formatter={(value, name, props) =>
                            [`${value.toFixed(2)} (${props.payload.label})`, 'Position']
                          }
                        />
                        <Line type="monotone" dataKey="position" stroke="#6b4caf" dot />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                  <div style={styles.suggestionsTitle}>Suggestions for Improvement</div>
                  <div style={styles.suggestionText}>{reportData.movementText}</div>
                </div>

                <div style={{ ...styles.breakdownContent, flexDirection: 'row', justifyContent: 'space-between' }}>
                  {/* Shoulder Posture Chart */}
                  <div style={{ ...styles.placeholderBox, flex: 1 }}>
                    <div style={styles.graphTitle}>Shoulder Posture Analysis</div>
                    <ResponsiveContainer width="100%" height={250}>
                      <PieChart>
                        <Pie
                          data={shoulderData}
                          dataKey="value"
                          nameKey="name"
                          outerRadius={60}
                          label={({ value }) => `${value}%`}
                        >
                          {shoulderData.map((entry, index) => (
                            <Cell key={`cell-shoulder-${index}`} fill={pieColors[index % pieColors.length]} />
                          ))}
                        </Pie>
                        <Tooltip formatter={(value) => `${value}%`} />
                        <Legend />
                      </PieChart>
                    </ResponsiveContainer>
                    <div style={styles.suggestionsTitle}>Suggestions for Improvement</div>
                    <div style={styles.suggestionText}>{reportData.shoulderText}</div>
                  </div>

                  {/* Hand Gesture Chart */}
                  <div style={{ ...styles.placeholderBox, flex: 1 }}>
                    <div style={styles.graphTitle}>Hand Gestures Analysis</div>
                    <ResponsiveContainer width="100%" height={250}>
                      <PieChart>
                        <Pie
                          data={handsData}
                          dataKey="value"
                          nameKey="name"
                          outerRadius={60}
                          label={({ value }) => `${value}%`}
                        >
                          {handsData.map((entry, index) => (
                            <Cell key={`cell-hands-${index}`} fill={pieColors[index % pieColors.length]} />
                          ))}
                        </Pie>
                        <Tooltip formatter={(value) => `${value}%`} />
                        <Legend />
                      </PieChart>
                    </ResponsiveContainer>
                    <div style={styles.suggestionsTitle}>Suggestions for Improvement</div>
                    <div style={styles.suggestionText}>{reportData.gestureText}</div>
                  </div>

                </div>
                {/* Placeholder Graph 3 */}
                <div style={styles.placeholderBox}>
                  <div style={styles.graphTitle}>Gaze Analysis</div>
                  <div style={{
                    position: 'relative',
                    width: '300px',
                    height: '300px',
                    margin: '20px auto'
                  }}>
                    {/* Purple human SVG background */}
                    <Mansvg style={{
                      position: 'absolute',
                      width: '70%',
                      height: '110%',
                      fill: '#6b4caf',
                      opacity: 1,
                      stroke: '#6b4caf',
                      left: '15%',
                      top: '-5%'
                    }} />

                    {/* Heatmap grid */}
                    <div style={{
                      position: 'absolute',
                      top: 0,
                      left: 0,
                      display: 'grid',
                      gridTemplateColumns: 'repeat(3, 1fr)',
                      gridTemplateRows: 'repeat(3, 1fr)',
                      width: '100%',
                      height: '100%',
                      zIndex: 1
                    }}>
                      {['up left', 'up', 'up right',
                        'left', 'center', 'right',
                        'down left', 'down', 'down right'].map((pos) => (
                          <div
                            key={pos}
                            onMouseEnter={(e) => e.currentTarget.style.transform = 'scale(1.1)'}
                            onMouseLeave={(e) => e.currentTarget.style.transform = 'scale(1)'}
                            style={{
                              backgroundColor: getHeatColor(gazePercentages[pos] || 0),
                              display: 'flex',
                              flexDirection: 'column',
                              alignItems: 'center',
                              justifyContent: 'center',
                              fontSize: '14px',
                              fontWeight: '600',
                              color: '#fff',
                              textShadow: '0 1px 3px rgba(0,0,0,0.8)',
                              position: 'relative',
                              cursor: 'pointer',
                              transition: 'transform 0.2s ease, background-color 0.3s ease',
                              borderRadius: '12px',
                              padding: '6px',
                              margin: '2px'
                            }}
                          >
                            <div style={{
                              fontSize: '16px',
                              fontWeight: '700'
                            }}>
                              {gazePercentages[pos]}%
                            </div>
                            <div style={{
                              fontSize: '12px',
                              marginTop: '2px'
                            }}>
                              {pos}
                            </div>
                          </div>
                        ))}
                    </div>
                  </div>
                  <div style={styles.suggestionsTitle}>Suggestions for Improvement</div>
                  <div style={styles.suggestionText}>{reportData.gazeText}</div>
                </div>
              </div>
            </div>

          </div>
        </div>
      </div>
      <div style={{ display: 'flex', justifyContent: 'center', marginTop: '20px' }}>
        <button className="download-button" onClick={handlePrint}>
          <FaDownload />
          Download Report
        </button>
      </div>
      <style>{`
.download-button {
  background-color: #6b4caf;
  color: white;
  border: none;
  padding: 12px 24px;
  border-radius: 8px;
  font-size: 16px;
  font-weight: 600;
  cursor: pointer;
  display: flex;
  justify-content: center;
  gap: 8px;
  transition: background-color 0.3s, transform 0.1s;
}

.download-button:hover {
  background-color: #5a3fa0; /* slightly darker purple on hover */
}

.download-button:active {
  transform: scale(0.95); /* slightly shrink on click */
}
`}</style>
    </div>
  );
}
