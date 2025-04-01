function updateMetrics() {
    const technique = document.getElementById('technique').value;

    // Handle redirection based on the selected technique
    if (technique === 'all') {
        window.location.href = "all_techniques.html";  // Redirect to All Techniques page
    } else if (technique === 'scoring_distribution_analysis') {
        window.location.href = "scoring_distribution_analysis.html";  // Redirect to Scoring & Distribution Analysis page
    } else if (technique === 'keywords_and_topics_analysis') {
        window.location.href = "keywords_and_topics_analysis.html";  // Redirect to Keywords and Topics Analysis page
    } else if (technique === 'deep_learning_based_analysis') {
        window.location.href = "deep_learning_based_analysis.html";  // Redirect to Deep Learning-Based Analysis page
    }
}
