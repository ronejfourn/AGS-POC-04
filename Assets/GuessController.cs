using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using Unity.Barracuda;

public class GuessController : MonoBehaviour
{
    [SerializeField]
    private DrawableRawImage DRI;
    [SerializeField]
    private NNModel onnxModel;

    [SerializeField]
    private RawImage inputPreview;
    [SerializeField]
    private Text outputText;

    private RenderTexture resizedTexture;
    private Texture2D inputTexture;

    private Model runtimeModel;
    private IWorker worker;
    private string outputLayerName;

    const int IMSIZE = 28;

    void Start()
    {
        runtimeModel = ModelLoader.Load(onnxModel);
        worker = WorkerFactory.CreateWorker(WorkerFactory.Type.Auto, runtimeModel);
        outputLayerName = runtimeModel.outputs[runtimeModel.outputs.Count - 1];

        resizedTexture = new RenderTexture(IMSIZE, IMSIZE, 0, RenderTextureFormat.R8);
        inputTexture = new Texture2D(IMSIZE, IMSIZE, TextureFormat.R8, false);
        inputPreview.texture = inputTexture;

        RenderTexture.active = resizedTexture;
        GL.Clear(true, true, DRI.clearColor);
        inputTexture.ReadPixels(new Rect(0, 0, IMSIZE, IMSIZE), 0, 0);
        RenderTexture.active = null;
        inputTexture.Apply();
    }

    public void Guess()
    {
        RenderTexture rt = DRI.drawCamera.targetTexture;
        Graphics.Blit(rt, resizedTexture);

        RenderTexture.active = resizedTexture;
        inputTexture.ReadPixels(new Rect(0, 0, IMSIZE, IMSIZE), 0, 0);
        RenderTexture.active = null;
        inputTexture.Apply();

        using Tensor ipTensor = new Tensor(inputTexture, 1);
        worker.Execute(ipTensor);
        Tensor opTensor = worker.PeekOutput(outputLayerName);

        float []predicted = opTensor.AsFloats();

        int m = 0;
        for (int i = 0; i < predicted.Length; i ++)
            if (predicted[i] > predicted[m])
                m = i;

        outputText.text = m.ToString();
    }

    private void OnDestroy()
    {
        worker?.Dispose();
    }
}
