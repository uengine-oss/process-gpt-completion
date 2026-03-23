/**
 * 이전에는 /audio-stream(TTS) 데모였습니다.
 * 음성 스트리밍은 별도 마이크로서비스로 이전되었습니다.
 */
document.addEventListener('DOMContentLoaded', function () {
    console.warn(
        '[audio-stream] /audio-stream 엔드포인트는 제거되었습니다. 전용 오디오 서비스를 사용하세요.'
    );
});
