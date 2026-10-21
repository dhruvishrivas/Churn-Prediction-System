
document.addEventListener("DOMContentLoaded", () => {
    const $ = (id) => document.getElementById(id);
    const form = $("predictionForm");
    const button = $("predictButton");

    if (!form || !button) {
        console.error("Prediction form or button not found.");
        return;
    }

    let distributionChart;
    let contractChart;

    const percent = (value) =>
        `${(Number(value) * 100).toFixed(1)}%`;

    // dahsboard ka kaam

    async function loadDashboard() {
        try {
            const response = await fetch("/api/dashboard");

            if (!response.ok) {
                throw new Error("Dashboard data unavailable.");
            }

            const data = await response.json();

            $("totalCustomers").textContent =
                Number(data.total_customers).toLocaleString("en-IN");

            $("churnRate").textContent = percent(data.churn_rate);

            $("highRiskCustomers").textContent =
                Number(data.high_risk_customers ?? 0)
                    .toLocaleString("en-IN");

            $("modelRecall").textContent = percent(data.recall);

            $("stayedPercentage").textContent =
                percent(1 - data.churn_rate);

            $("churnedPercentage").textContent =
                percent(data.churn_rate);

            renderCharts(data);
            renderMetrics(data.metrics || {});
        } catch (error) {
            console.error("Dashboard:", error.message);
        }
    }

    function renderCharts(data) {
        if (typeof Chart === "undefined") return;

        const distributionCanvas = $("churnDistributionChart");
        const contractCanvas = $("contractChurnChart");

        if (distributionCanvas) {
            if (distributionChart) distributionChart.destroy();

            distributionChart = new Chart(distributionCanvas, {
                type: "doughnut",
                data: {
                    labels: ["Stayed", "Churned"],
                    datasets: [{
                        data: [
                            data.stayed_count,
                            data.churned_count
                        ],
                        backgroundColor: ["#218b70", "#d45f68"],
                        borderWidth: 0,
                        hoverOffset: 5
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    cutout: "75%",
                    plugins: {
                        legend: { display: false }
                    }
                }
            });
        }

        if (contractCanvas) {
            if (contractChart) contractChart.destroy();

            const contracts = data.contract_churn || {};

            contractChart = new Chart(contractCanvas, {
                type: "bar",
                data: {
                    labels: Object.keys(contracts),
                    datasets: [{
                        label: "Churn rate",
                        data: Object.values(contracts),
                        backgroundColor: [
                            "#8193ec",
                            "#5267d9",
                            "#9eafff"
                        ],
                        borderRadius: 6,
                        maxBarThickness: 45
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            callbacks: {
                                label: (item) =>
                                    `Churn rate: ${percent(item.raw)}`
                            }
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 1,
                            ticks: {
                                callback: (value) => `${value * 100}%`
                            },
                            grid: { color: "#edf0f6" }
                        },
                        x: {
                            grid: { display: false }
                        }
                    }
                }
            });
        }
    }

    function renderMetrics(metrics) {
        const fields = {
            accuracy: "modelAccuracy",
            precision: "modelPrecision",
            recall: "performanceRecall",
            f1_score: "modelF1"
        };

        Object.entries(fields).forEach(([key, elementId]) => {
            const element = $(elementId);

            if (!element || metrics[key] == null) return;

            element.textContent = percent(metrics[key]);
        });
    }

    

    const tenureInput = form.elements.namedItem("tenure");
    const monthlyInput = form.elements.namedItem("MonthlyCharges");
    const totalInput = form.elements.namedItem("TotalCharges");

    function updateTotalCharges() {
        if (!tenureInput || !monthlyInput || !totalInput) return;

        const tenure = Number(tenureInput.value);
        const monthlyCharges = Number(monthlyInput.value);

        if (
            tenureInput.value === "" ||
            monthlyInput.value === "" ||
            !Number.isFinite(tenure) ||
            !Number.isFinite(monthlyCharges) ||
            tenure < 0 ||
            monthlyCharges < 0
        ) {
            totalInput.value = "";
            return;
        }

        // total vallue agr mintky charge hai toh 
        const estimatedTotal = monthlyCharges * tenure;

        totalInput.value = estimatedTotal.toFixed(2);
    }

    // total value change hogi dusri vales se
    if (totalInput) {
        totalInput.readOnly = true;
    }

    if (tenureInput) {
        tenureInput.addEventListener("input", updateTotalCharges);
    }

    if (monthlyInput) {
        monthlyInput.addEventListener("change", updateTotalCharges);
        monthlyInput.addEventListener("input", updateTotalCharges);
    }

    updateTotalCharges();

    // --idhr churn predict 

    form.addEventListener("submit", async (event) => {
        event.preventDefault();

        if (!form.reportValidity()) return;

        button.disabled = true;
        button.textContent = "Analyzing...";

        try {
            updateTotalCharges();

            const customer = Object.fromEntries(
                new FormData(form).entries()
            );

            ["tenure", "MonthlyCharges", "TotalCharges"].forEach(
                (key) => {
                    customer[key] = Number(customer[key]);
                }
            );

            const response = await fetch("/api/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(customer)
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(
                    result.error || "Prediction failed."
                );
            }

            const risk = Number(result.churn_probability);

            if (
                !Number.isFinite(risk) ||
                risk < 0 ||
                risk > 1
            ) {
                throw new Error(
                    "Invalid prediction returned by server."
                );
            }

            $("predictionResult").hidden = false;
            $("resultStatus").textContent = result.risk_level;
            $("riskPercentage").textContent = percent(risk);

            $("resultMessage").textContent =
                result.prediction === 1
                    ? "This customer is predicted to churn."
                    : "This customer is predicted to stay.";

            $("riskBar").style.width = `${risk * 100}%`;

            $("retentionRecommendation").textContent =
                result.recommendation ||
                "Review the customer's service experience and retention options.";

            $("predictionResult").scrollIntoView({
                behavior: "smooth",
                block: "nearest"
            });

        } catch (error) {
            console.error("Prediction:", error);
            alert(error.message);
        } finally {
            button.disabled = false;
            button.textContent = "Predict Churn Risk →";
        }
    });

    form.addEventListener("reset", () => {
        $("predictionResult").hidden = true;

      
        setTimeout(updateTotalCharges, 0);
    });

    loadDashboard();
});
