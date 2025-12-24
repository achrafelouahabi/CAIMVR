def get_default_config(data_name):
    if data_name in ['Caltech101-20']:
        return dict(
            seed=4,
            view=2,
            training=dict(
                pretrain_epochs=150,
                batch_size=256,
                epoch=300,
                alpha=10,
                lambda2=0.1,
                lambda1=0.11,
                lr=1.0e-4,
            ),
            Autoencoder=dict(
                arch1=[1984, 1024, 1024, 1024, 128],
                arch2=[512, 1024, 1024, 1024, 128],
                activations1='gelu',
                activations2='gelu',
                batchnorm=True,
                heads=16
            ),
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
                activations1='gelu',
                activations2='gelu',
                heads=8
            )
 
        )

    elif data_name in ['Scene_15']:
        """The default configs."""
        return dict(
            seed=7,
            view=2,
            k=15,
            training=dict(
                lr=1.0e-3,
                pretrain_epochs=50,
                batch_size=1024,
                epoch=500,
                alpha=10,
                lambda2=0.1,
                lambda1=0.11,
            
            ),
            Autoencoder=dict(
                arch1=[40, 1024, 1024, 1024, 128],
                arch2=[59, 1024, 1024, 1024, 128],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
                heads=16
            ),
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
                activations1='relu',
                activations2='relu',
                heads=4
            ) )


    elif data_name in ['hand']:
        """The default configs."""
        return dict(
            seed=5,
            view=2,
            k=5,
            training=dict(
                lr=1.0e-4,
                pretrain_epochs=0,
                batch_size=256,
                epoch=200,
                alpha=10,
                lambda2=0.1,
                lambda1=0.11,
            
            ),
            Autoencoder=dict(
                arch1=[240, 1024, 1024, 1024, 128],
                arch2=[216, 1024, 1024, 1024, 128],
                activations1='gelu',
                activations2='gelu',
                batchnorm=True,
                heads=16
            ),
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
                activations1='gelu',
                activations2='gelu',
                heads=8
            ) )
    elif data_name in ['NoisyMNIST']:
        """The default configs."""
        return dict(
            seed=1,
            view=2,
            Autoencoder=dict(
                arch1=[784, 1024, 1024, 1024, 40],
                arch2=[784, 1024, 1024, 1024, 40],
                activations1='gelu',
                activations2='gelu',
                batchnorm=True,
                heads=12
            ),
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
                activations1='gelu',
                activations2='gelu',
                heads=14
            ),
            training=dict(
                lr=1.0e-4,
                pretrain_epochs=0,
                batch_size=256,
                epoch=500,
                alpha=10,
                lambda1=0.11,
                lambda2=0.1,
            ),
        )

    elif data_name in ['LandUse_21']:
        """The default configs."""
        return dict(
            seed=2,
            view=2,
            Autoencoder=dict(
                arch1=[59, 1024, 1024, 1024, 40],
                arch2=[40, 1024, 1024, 1024, 40],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
                heads=6
            ),
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
                activations1='relu',
                activations2='relu',
                heads=16
            ),
            training=dict(
                lr=5.0e-4,
                pretrain_epochs=0,
                batch_size=256,
                epoch=500,
                epoch_3=500,
                alpha=10,
                lambda2=0.1,
                lambda1=0.11,
            ),
        )

    elif data_name in ['MSRC_v1']:
        return dict(
            missing_rate=0,
            
            Prediction=dict(
                arch1=[64, 64, 64],  
                arch2=[64, 64, 64],  
                activations1='gelu',
                activations2='gelu',
                heads=1
            ),
            
            Autoencoder=dict(
                arch1=[512, 156],     
                arch2=[256, 156],
             
                activations1='gelu',
                activations2='gelu',
                batchnorm=True,
                heads=1
            ),
            
            training=dict(
                lr=1e-3,                    
                pretrain_epochs=50, 
                batch_size=34,
                epoch=300, 
                epoch_3=500,
                alpha=10.0,
                lambda1=0.11,
                lambda2=0.1,
            ),
            view=2,
            seed=5,
        )



    else:
        raise Exception('Undefined data_name')
