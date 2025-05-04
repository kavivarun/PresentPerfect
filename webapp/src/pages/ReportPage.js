import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  LineChart, Line, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, Tooltip, CartesianGrid, Legend
} from 'recharts';


const COLORS = ['#5D2E8C', '#B288C0', '#8884d8', '#82ca9d', '#ffc658', '#ff7300'];

export default function ReportPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const reportData = location.state?.reportData;

  if (!reportData) {
    navigate('/');
    return null;
  }

  // Transform emotion and gaze data into array format
  const chartData = Object.keys(reportData.emotion).map(sec => ({
    second: parseInt(sec),
    emotion: reportData.emotion[sec],
    gaze: reportData.gaze[sec],
    horizontal: reportData.horizontal[sec],
    tilt: reportData.tilt[sec],
    hunch: reportData.hunch[sec],
  }));

  // Count frequency for pie charts
  const emotionCounts = countBy(reportData.emotion);
  const gazeCounts = countBy(reportData.gaze);

  return (
    <div className="p-6 bg-gray-100 min-h-screen">
      <h1 className="text-3xl font-bold text-center mb-6 text-purple-800">Your Presentation Analysis Report</h1>

      {/* Emotion over time */}
      <ReportCard title="Emotion Over Time">
        <BarChart width={600} height={300} data={chartData}>
          <XAxis dataKey="second" />
          <YAxis />
          <Tooltip />
          <CartesianGrid stroke="#eee" />
          <Bar dataKey="emotion" fill="#5D2E8C" />
        </BarChart>
      </ReportCard>

      {/* Emotion distribution */}
      <ReportCard title="Emotion Distribution">
        <PieChart width={400} height={300}>
          <Pie
            data={Object.entries(emotionCounts).map(([name, value]) => ({ name, value }))}
            dataKey="value"
            nameKey="name"
            cx="50%"
            cy="50%"
            outerRadius={100}
            label
          >
            {Object.keys(emotionCounts).map((_, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
            ))}
          </Pie>
          <Legend />
        </PieChart>
      </ReportCard>

      {/* Gaze over time */}
      <ReportCard title="Gaze Over Time">
        <BarChart width={600} height={300} data={chartData}>
          <XAxis dataKey="second" />
          <YAxis />
          <Tooltip />
          <CartesianGrid stroke="#eee" />
          <Bar dataKey="gaze" fill="#B288C0" />
        </BarChart>
      </ReportCard>

      {/* Gaze distribution */}
      <ReportCard title="Gaze Distribution">
        <PieChart width={400} height={300}>
          <Pie
            data={Object.entries(gazeCounts).map(([name, value]) => ({ name, value }))}
            dataKey="value"
            nameKey="name"
            cx="50%"
            cy="50%"
            outerRadius={100}
            label
          >
            {Object.keys(gazeCounts).map((_, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
            ))}
          </Pie>
          <Legend />
        </PieChart>
      </ReportCard>

      {/* Side tilt and hunch over time */}
      <ReportCard title="Posture Metrics Over Time">
        <LineChart width={600} height={300} data={chartData}>
          <XAxis dataKey="second" />
          <YAxis />
          <Tooltip />
          <CartesianGrid stroke="#eee" />
          <Line type="monotone" dataKey="tilt" stroke="#82ca9d" name="Tilt (°)" />
          <Line type="monotone" dataKey="hunch" stroke="#ffc658" name="Hunch (°)" />
        </LineChart>
      </ReportCard>

      <div className="text-center mt-8">
        <button
          onClick={() => window.print()}
          className="bg-purple-700 text-white px-6 py-2 rounded-full hover:bg-purple-800 transition"
        >
          Download Report as PDF
        </button>
      </div>
    </div>
  );
}

// Helper card wrapper
function ReportCard({ title, children }) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6 mb-8 mx-auto max-w-4xl">
        <h2 className="text-xl text-purple-700 mb-4">{title}</h2>
        <div className="flex justify-center">{children}</div>
      </div>
    );
  }

// Helper to count occurrences
function countBy(obj) {
  return Object.values(obj).reduce((acc, val) => {
    acc[val] = (acc[val] || 0) + 1;
    return acc;
  }, {});
}
