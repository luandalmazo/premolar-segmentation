# Segmentação de dentes pré-molares através de técnicas de pré-processamento de imagens e Deep Learning

Este projeto tem como objetivo a segmentação automatizada de dentes pré-molares superiores (14, 15, 24 e 25) utilizando imagens de Tomografia Computadorizada de Feixe Cônico (CBCT).

---

## 🦷 Para Dentistas (Tutorial 3D Slicer)

Se o seu objetivo é utilizar o modelo já treinado para auxiliar em segmentações, siga o passo a passo abaixo para rodar a segmentação automática.

### Passo a Passo:
1. **Instalação do Slicer:** Certifique-se de ter o [3D Slicer](https://download.slicer.org/) instalado em seu computador.
2. **Instalação da Extensão:** - Vá em `View` --> `Extensions Manager` --> `Install Extensions`. 
   - Procure por **nnUNet** e clique em instalar. Reinicie o Slicer se solicitado.
3. **Configuração do Módulo:** - No menu de módulos (aba `Welcome to Slicer`), vá na categoria `Segmentation` e selecione **nnUNet**. 
   - Clique em `nnUNet Install` e depois em `Install` para instalar as dependências internas.
4. **Execução da Segmentação:**
   - Realize download dos arquivos disponibilizados [AQUI para nnU-Net](https://drive.google.com/drive/folders/1p40SNDsTysR7Bkl3_OfqckU2fip1h1Sc?usp=sharing) ou [AQUI para nnU-Net ResNet](https://drive.google.com/drive/folders/10wGI0_L4LN5kStCGFCX55olkQfJUY64o?usp=sharing) (Realize o download da pasta completa).
   - Em `nnUNet Run Settings`, no campo `Model path`, aponte para a pasta que você realizou o download.
   - No campo `Checkpoint name`, coloque o nome `checkpoint_best.pth`.
   - Para `Folds`, coloque o valor `0`.
   - Importe o volume da tomografia para o 3D Slicer (arraste o arquivo DICOM ou .nii.gz).
   - No campo `Input volume`, selecione o volume que foi importado e pressione **Apply**.
---

## 💻 Para Desenvolvedores

Este guia descreve os passos necessários para replicar o processamento, treinamento e avaliação realizados.

**Instale as dependências**

```bash
pip install -r requirements.txt
```

**Instale a nnU-Net**

```bash
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .
```


### 1. Preparação e Aumento de Dados

Antes de iniciar o treinamento, o conjunto de dados deve ser convertido para o formato padrão da nnU-Net.

#### Conversão do Dataset

Certifique-se de que os dados brutos estão organizados e execute a conversão para o padrão `Dataset903_Tooth`:

    python convert_dataset.py

#### Data Augmentation Manual

Execute o script para gerar o aumento de 11x nos volumes:

    python manual_augmentation.py

#### Geração de Splits

Para garantir a reprodutibilidade da validação cruzada, gere os arquivos de divisão dos folds:

    python generate_splits.py

### 2. Planejamento e Pré-processamento

O planejamento extrai as propriedades do dataset para configurar as redes.

#### nnU-Net

    # Planejamento padrão da nnU-Net v2 para 80GB de VRAM
    nnUNetv2_plan_and_preprocess -d 903 -c 3d_fullres

#### nnU-Net ResNet (Caso as imagens já tenham sido preprocessadas anteriormente para treinamento da nnU-Net nativa)

    nnUNetv2_plan_experiment -d 903 \
        -pl nnUNetPlannerResEncL \
        -gpu_memory_target 80 \
        -overwrite_plans_name nnUNetResEncUNetPlans_L_80G

### 3. Treinamento das Arquiteturas

Como o dataset já passou por um processo de aumento de dados manual, utilizamos o trainer `nnUNetTrainerNoDA` para desativar as transformações nativas redundantes.

#### nnU-Net Nativa

    # Treinamento do Fold 0 (Repetir de 0 a 4)
    nnUNetv2_train 903 3d_fullres 0 -tr nnUNetTrainerNoDA

#### nnU-Net ResNet 

Para a variante ResNet, certifique-se de que o plano específico para o
encoder residual foi gerado:

    # Treinamento do Fold 0 utilizando a arquitetura ResNet
    nnUNetv2_train 903 3d_fullres 0 -p nnUNetResEncUNetPlans_L_80G -tr nnUNetTrainerNoDA

### 4. Predição e Avaliação

Após a conclusão do treinamento, siga os passos para validar os modelos nos dados de teste.

#### Organização dos Testes

Prepare os diretórios de saída:

    python organize_test_folders.py

#### Predição

Execute a inferência utilizando o melhor checkpoint:

    # nnU-Net nativa
    
    nnUNetv2_predict -i [DIRETORIO_ENTRADA] -o [DIRETORIO_SAIDA] -d 903 -c 3d_fullres -f [FOLDS] -tr [TRAINER_UTILIZADO]
    
    # nnU-Net ResNet
    
    nnUNetv2_predict -i [DIRETORIO_ENTRADA] -o [DIRETORIO_SAIDA] -d 903 -c 3d_fullres -f [FOLDS] -tr [TRAINER_UTILIZADO]-p nnUNetResEncUNetPlans_L_80G
    

#### Geração de Métricas

Extraia os resultados quantitativos (Dice, IoU, HD95, ASSD):

    python3 generate_test_results.py
    python3 analyze_results.py

### 🎥 Demonstração no 3D Slicer

Caso tenha interesse em verificar o comportamento do modelo integrado em
ambiente clínico simulado, assista ao nosso vídeo demonstrativo:

👉 Assista ao vídeo [AQUI](https://drive.google.com/file/d/1wNqsYPZWvQr9I3iy7mbz6cN8zq6dLlKF/view?usp=sharing)