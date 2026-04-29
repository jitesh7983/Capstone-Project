import React from 'react';
import { Bar } from 'react-chartjs-2';

const Analytics = ({ metrics, matrixData }) => {
  // Chart Configuration
  const chartData = {
    labels: metrics.map((m) => m.model),
    datasets: [
      {
        label: "Accuracy Score",
        data: metrics.map((m) => m.accuracy),
        backgroundColor: "rgba(75, 192, 192, 0.6)",
      },
      {
        label: "F1 Score",
        data: metrics.map((m) => m.f1_score),
        backgroundColor: "rgba(153, 102, 255, 0.6)",
      }
    ]
  };

  return (
    <div style={{ marginTop: '30px', textAlign: 'left' }}>
      <h2 style={{ color: '#2c3e50', borderBottom: '2px solid #eee', paddingBottom: '10px' }}>
        📊 Model Performance Metrics
      </h2>
      
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginTop: '20px' }}>
        {/* Metric Bar Chart */}
        <div style={{ background: '#fff', padding: '15px', borderRadius: '10px', boxShadow: '0 2px 10px rgba(0,0,0,0.05)' }}>
          <h4>Algorithm Comparison</h4>
          <Bar data={chartData} />
        </div>

        {/* Confusion Matrix Table */}
        <div style={{ background: '#fff', padding: '15px', borderRadius: '10px', boxShadow: '0 2px 10px rgba(0,0,0,0.05)' }}>
          <h4>Confusion Matrix (Heatmap)</h4>
          <div style={{ display: 'grid', gridTemplateColumns: `repeat(${matrixData.labels.length + 1}, 1fr)`, gap: '5px' }}>
            <div style={{ fontWeight: 'bold', fontSize: '10px' }}>Actual \ Pred</div>
            {matrixData.labels.map(label => (
              <div key={label} style={{ fontWeight: 'bold', fontSize: '10px', textAlign: 'center' }}>{label}</div>
            ))}
            {matrixData.values.map((row, i) => (
              <React.Fragment key={i}>
                <div style={{ fontWeight: 'bold', fontSize: '10px' }}>{matrixData.labels[i]}</div>
                {row.map((val, j) => (
                  <div key={j} style={{ 
                    background: i === j ? '#d4edda' : '#f8d7da', 
                    textAlign: 'center', 
                    padding: '5px',
                    fontSize: '12px',
                    borderRadius: '3px'
                  }}>
                    {val}
                  </div>
                ))}
              </React.Fragment>
            ))}
          </div>
          <p style={{ fontSize: '11px', color: '#666', marginTop: '10px' }}>
            *Green cells show correct predictions, red shows misclassifications.
          </p>
        </div>
      </div>
    </div>
  );
};

export default Analytics;