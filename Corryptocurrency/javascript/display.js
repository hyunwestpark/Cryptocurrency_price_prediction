var showVal = false;

const actualPrices = [100, 150, 200, 180, 220];

// 예측된 코인 가격 데이터
const predictedPrices = [120, 160, 190, 170, 210];

function onDisplay() {
  if (showVal) {
    return;
  } else {
    Corrypto = [{ name: "Ethereum" }, { name: "Ripple" }, { name: "Stellar" }];
    var td = document.createElement("td");
    var ul = document.createElement("ul");
    Corrypto.forEach((item) => {
      var li = document.createElement("li");
      li.appendChild(document.createTextNode(item.name));
      ul.appendChild(li);
    });
    showVal = true;
  }
  td.appendChild(ul);
  document.getElementById("1").appendChild(td);
  createChart();
}

// 그래프 생성 및 표시
function createChart() {
  var td = document.createElement("td");
  var cvs = document.createElement("canvas");
  const ctx = document.getElementById("coinChart").getContext("2d");
  new Chart(ctx, {
    type: "line",
    data: {
      labels: [
        "2023-01-01",
        "2023-01-02",
        "2023-01-03",
        "2023-01-04",
        "2023-01-05",
      ],
      datasets: [
        {
          label: "Actual Price",
          data: actualPrices,
          borderColor: "blue",
          fill: false,
        },
        {
          label: "Predicted Price",
          data: predictedPrices,
          borderColor: "green",
          fill: false,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false, // 사이즈 조절 옵션
      scales: {
        x: {
          display: true,
          title: {
            display: true,
            text: "Date",
          },
        },
        y: {
          display: true,
          title: {
            display: true,
            text: "Price",
          },
        },
      },
    },
  });
}

// 그래프 생성 호출
