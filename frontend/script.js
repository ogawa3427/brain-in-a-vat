document.getElementById('imageInput').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        // プレビューを表示
        const reader = new FileReader();
        reader.onload = function(e) {
            const preview = document.getElementById('imagePreview');
            preview.src = e.target.result;
            preview.style.display = 'block';
        }
        reader.readAsDataURL(file);

        // キャプション生成
        generateCaption(file);
    }
});

async function generateCaption(file) {
    const loading = document.getElementById('loading');
    const captionDiv = document.getElementById('caption');
    
    loading.style.display = 'block';
    captionDiv.textContent = '';

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/generate-caption', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        if (data.error) {
            throw new Error(data.error);
        }
        
        captionDiv.textContent = data.caption;
    } catch (error) {
        captionDiv.textContent = 'エラーが発生しました: ' + error.message;
    } finally {
        loading.style.display = 'none';
    }
}