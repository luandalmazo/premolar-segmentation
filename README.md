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
   - Realize download dos arquivos disponibilizados [AQUI](https://drive.google.com/drive/folders/1p40SNDsTysR7Bkl3_OfqckU2fip1h1Sc?usp=sharing) (Realize o download da pasta completa).
   - Em `nnUNet Run Settings`, no campo `Model path`, aponte para a pasta que você realizou o download.
   - No campo `Checkpoint name`, coloque o nome `checkpoint_best.pth`.
   - Para `Folds`, coloque o valor `0`.
   - Importe o volume da tomografia para o 3D Slicer (arraste o arquivo DICOM ou .nii.gz).
   - No campo `Input volume`, selecione o volume que foi importado e pressione **Apply**.
---

## 💻 Para Desenvolvedores (Reprodução dos Resultados)

Este guia descreve os passos necessários para replicar o treinamento e os testes realizados no cluster de alto desempenho.

### 1. Preparação dos Dados
Os dados devem seguir o padrão `Dataset903_Tooth` da nnU-Net. As máscaras de segmentação devem ser rotuladas de 1 a 4 para os dentes pré-molares e 0 para o fundo (background).

### 2. Planejamento e Treinamento (nnU-Net Nativa)
Para reproduzir os resultados utilizando a arquitetura padrão da nnU-Net, siga os comandos abaixo. Como o dataset já passou por um processo de aumento de dados manual de 11x, utilizamos o trainer `nnUNetTrainerNoDA` para desativar as transformações nativas durante o treino.

```bash
# Planejamento padrão da nnU-Net v2 para 80GB de VRAM
nnUNetv2_plan_and_preprocess -d 903 -c 3d_fullres 

# Iniciar o treinamento (Fold 0) com o trainer sem Data Augmentation
nnUNetv2_train 903 3d_fullres 0 -tr nnUNetTrainerNoDA