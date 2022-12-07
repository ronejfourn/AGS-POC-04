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
        inputPreview.texture = resizedTexture;
        RenderTexture.active = resizedTexture;
        GL.Clear(true, true, DRI.clearColor);
        RenderTexture.active = null;
    }

    public void Guess()
    {
        RenderTexture rt = DRI.drawCamera.targetTexture;
        Graphics.Blit(rt, resizedTexture);

        using Tensor ipTensor = new Tensor(resizedTexture, 1);
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
