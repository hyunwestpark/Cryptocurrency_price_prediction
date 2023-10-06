function createChart() {
  fetch("data/data0.json")
    .then((response) => response.json())
    .then((data) => {
      const coinData = JSON.parse(data);
      const labels = coinData.map((item) => item.time);
      const prices = coinData.map((item) => item.price);

      const ctx = document.getElementById("coinChart").getContext("2d");
      new Chart(ctx, {
        type: "line",
        data: {
          labels: labels,
          datasets: [
            {
              label: "Price",
              data: prices,
              backgroundColor: "rgba(0, 123, 255, 0.2)", // Optional: Set a background color
              borderColor: "rgba(0, 123, 255, 1)", // Optional: Set a border color
              borderWidth: 1, // Optional: Set a border width
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
    })
    .catch((error) => {
      console.error("Error:", error);
    });
}

// const actualPrices = coin.price;
// console.log(actualPrices);

// // 예측된 코인 가격 데이터
// const predictedPrices = [120, 160, 190, 170, 210];

// // 그래프 생성 및 표시
// function createChart() {
//   const ctx = document.getElementById("coinChart").getContext("2d");
//   new Chart(ctx, {
//     type: "line",
//     data: {
//       labels: [
//         "2023-01-01",
//         "2023-01-02",
//         "2023-01-03",
//         "2023-01-04",
//         "2023-01-05",
//       ],
//       datasets: [
//         {
//           label: "Actual Price",
//           data: actualPrices,
//           borderColor: "blue",
//           fill: false,
//         },
//         {
//           label: "Predicted Price",
//           data: predictedPrices,
//           borderColor: "green",
//           fill: false,
//         },
//       ],
//     },
//     options: {
//       responsive: true,
//       maintainAspectRatio: false, // 사이즈 조절 옵션
//       scales: {
//         x: {
//           display: true,
//           title: {
//             display: true,
//             text: "Date",
//           },
//         },
//         y: {
//           display: true,
//           title: {
//             display: true,
//             text: "Price",
//           },
//         },
//       },
//     },
//   });
// }

// 그래프 생성 호출
createChart();
