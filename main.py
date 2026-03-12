import os
import argparse
import subprocess
import sys

def run_command(cmd, description):
    print(f"\n{'='*60}")
    print(f"🚀 실행 중: {description}")
    print(f"명령어: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    # Run the command and stream output directly to console
    process = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
    process.communicate()
    
    if process.returncode != 0:
        print(f"\n❌ 에러 발생: {description} 단계에서 실패했습니다 (코드: {process.returncode}). 프로그램을 중단합니다.")
        sys.exit(process.returncode)
        
    print(f"\n✅ 완료: {description}\n")

def main():
    parser = argparse.ArgumentParser(description="구조물 안정성 예측 AI - 원클릭 실행 스크립트")
    parser.add_argument("--data_dir", type=str, default="./open", help="압축 해제된 데이터 폴더 경로")
    parser.add_argument("--batch_size", type=int, default=16, help="GPU 메모리에 맞게 조절하세요 (예: 8, 16, 32)")
    parser.add_argument("--epochs", type=int, default=30, help="학습 에폭 수 (기본: 30)")
    parser.add_argument("--teacher_backbone", type=str, default="timesformer_base_patch16_224", help="비디오 Teacher 모델 백본")
    parser.add_argument("--student_backbone", type=str, default="convnext_tiny", help="제출용 Student 모델 백본")
    parser.add_argument("--output", type=str, default="final_submission.csv", help="최종 생성될 제출 파일 이름")
    
    args = parser.parse_args()
    
    # Ensure current directory has the necessary scripts
    scripts = ["train_video_teacher.py", "generate_pseudo_labels.py", "train_student.py", "inference.py"]
    for s in scripts:
        if not os.path.exists(s):
            print(f"에러: {s} 파일이 현재 폴더에 없습니다. 모든 스크립트가 같은 위치에 있어야 합니다.")
            sys.exit(1)
            
    if not os.path.exists(args.data_dir):
        print(f"에러: 데이터 폴더 '{args.data_dir}'를 찾을 수 없습니다. open.zip을 해제하여 준비해주세요.")
        sys.exit(1)
        
    print("\n단 한 번의 실행으로 전체 파이프라인(Teacher 학습 -> 라벨 추출 -> Student 학습 -> 추론)을 시작합니다.\n")
    
    # --- STEP 1: Train Video Teacher ---
    teacher_weights = os.path.join("checkpoints", f"best_teacher_{args.teacher_backbone}.pth")
    if os.path.exists(teacher_weights):
         print(f"정보: 기존에 학습된 Teacher 체크포인트({teacher_weights})가 존재하여, Step 1을 건너뜁니다.")
    else:
        run_command([
            sys.executable, "train_video_teacher.py",
            "--data_dir", args.data_dir,
            "--batch_size", str(args.batch_size // 2 if args.batch_size > 4 else 2), # Video models use more memory
            "--epochs", "10",
            "--backbone", args.teacher_backbone
        ], "Step 1: Video Teacher 모델 학습 (동역학 사전학습)")
        
    # --- STEP 2: Generate Pseudo Labels ---
    soft_labels_file = "teacher_soft_labels.csv"
    if os.path.exists(soft_labels_file):
        print(f"정보: 기존에 추출된 소프트 라벨({soft_labels_file})이 존재하여, Step 2를 건너뜁니다.")
    else:
        run_command([
            sys.executable, "generate_pseudo_labels.py",
            "--data_dir", args.data_dir,
            "--weights_path", teacher_weights,
            "--output_file", soft_labels_file
        ], "Step 2: Train 데이터에 대한 정밀한 소프트 확률표(Pseudo-label) 생성")

    # --- STEP 3: Train Student Model with Distillation ---
    student_weights = os.path.join("checkpoints", f"best_student_{args.student_backbone}.pth")
    run_command([
        sys.executable, "train_student.py",
        "--data_dir", args.data_dir,
        "--teacher_preds", soft_labels_file,
        "--batch_size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--backbone", args.student_backbone
    ], "Step 3: 지식 증류(Distillation) 기반 최종 Student(이미지 2장) 모델 강력 학습")

    # --- STEP 4: Inference and Generation of Submission csv ---
    run_command([
        sys.executable, "inference.py",
        "--data_dir", args.data_dir,
        "--weights_path", student_weights,
        "--backbone", args.student_backbone,
        "--advanced", # Crucial to use the AdvancedStructureModel
        "--output_file", args.output,
        "--batch_size", str(args.batch_size * 2) # Inference can handle larger batches
    ], f"Step 4: 최종 데이콘 제출 파일({args.output}) 생성")
    
    print(f"\n🎉 축하합니다! 모든 파이프라인이 완료되었습니다.")
    print(f"   폴더 안에 있는 [{args.output}] 파일을 데이콘에 제출하세요.")

if __name__ == "__main__":
    main()
