## Team 6 Brownie
|항목|내용|
|---|---|
|프로젝트명|대형 CLIP 모델의 제로샷 일반화 능력 유지를 위해 이미지-텍스트 쌍의 대조적 임베딩 관계 구조를 그대로 전이하는 경량 멀티모달 모델 지식 증류 기법 제안|
|프로젝트 키워드|Multi-modal AI(CLIP), 경량화(knowledge distillation), Loss Function, TinyCLIP|
|트랙|연구 트랙|
|프로젝트멤버|양인경, 장한나, 정진이|
|팀지도교수|심재형|
|무엇을 만들고자 하는가|멀티모달 모델인 CLIP을 중심으로, 자원이 제한되는 환경에서도 성능은 유지할 수 있도록, knowledge distillation 기법을 적용하여 모델을 경량화하고자 합니다.|
|고객|AI/ML 기술을 사용하는 스타트업 또는 기업: 데이터 용량이 제한적이거나, 실시간 응답 속도가 중요한 서비스를 개발하는 기업들은 가벼우면서도 성능이 우수한 모델이 필요로 합니다. 기존의 거대 Transformer 기반 멀티모달 모델은 높은 연산 비용과 느린 추론 속도 때문에 사용하기 어려워, 연구를 통해 빠르고 가벼운 CLIP 모델 제공하고자 합니다.|
|Pain Point|최근 텍스트, 이미지, 음성 등 여러 종류의 데이터를 동시에 이해하고 처리하는 멀티모달(Multi-modal) 모델이 AI 기술의 새로운 핵심으로 부상하고 있습니다. 하지만 멀티모달 모델의 특성상 여러 인코더와 이를 통합하는 복잡한 아키텍처를 가지고, 이는 단일 모델보다 훨씬 더 큰 모델 크기와 많은 연산량을 요구합니다. 이는 곧 배포의 한계, 비용 부담, 환경 문제 등으로 이어지며 기술의 대중화와 실용화를 막는 걸림돌이 되고 있습니다.|
|사용할 소프트웨어 패키지의 명칭과 핵심기능/용도, 사용시나리오| - pyTorch: 모델 학습 및 지식 증류 구현.<br>- CLIP : 멀티모달 모델. 지식 증류 과정에서의 교사 모델.<br>- Knowledge Distillation : KD Loss 함수 및 구현 참고.<br>- TinyCLIP: 지식 증류 과정에서의 학생 모델.<br>- Transformer: Text tokenization.<br>- SCIKIT-LEARN: zero-shot 평가 및 시각화.|
|사용할 소프트웨어 패키지의 명칭과 URL| - pyTorch: https://pytorch.org<br>- CLIP : https://github.com/openai/CLIP<br>- Knowledge Distillation : https://tutorials.pytorch.kr/beginner/knowledge_distillation_tutorial.html<br>- TinyCLIP: https://github.com/microsoft/Cream/tree/main/TinyCLIP<br>- Transformer: https://github.com/huggingface/transformers<br>- SCIKIT-LEARN: https://scikit-learn.org/stable/|
|팀그라운드룰|https://github.com/TeamBrowniewha/TeamBrownie/blob/main/GroundRule.MD|
|최종수정일|2025.10.13|
